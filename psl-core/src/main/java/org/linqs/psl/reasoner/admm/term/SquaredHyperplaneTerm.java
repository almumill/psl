/*
 * This file is part of the PSL software.
 * Copyright 2011-2015 University of Maryland
 * Copyright 2013-2020 The Regents of the University of California
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.linqs.psl.reasoner.admm.term;

import org.linqs.psl.model.rule.GroundRule;
import org.linqs.psl.model.rule.WeightedGroundRule;
import org.linqs.psl.reasoner.term.Hyperplane;
import org.linqs.psl.util.FloatMatrix;
import org.linqs.psl.util.HashCode;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Semaphore;

/**
 * Objective term for an ADMMReasoner that is based on a squared hyperplane in some way.
 *
 * This general class covers two specific types of terms:
 * 1) SquaredHingeLossTerm: weight * [max(0, coefficients^T * y - constant)]^2.
 * 2) SquaredLinearLossTerm: weight * [coefficients^T * x - constant]^2
 * Where y can be either local or concensus values.
 *
 * The reason these terms are housed in a single class instead of subclasses is for performance
 * in streaming settings where terms must be quickly serialized and deserialized.
 *
 * All coefficients must be non-zero.
 */
public class SquaredHyperplaneTerm extends ADMMObjectiveTerm {
    /**
     * The specific type of term represented by this instance.
     */
    public static enum HyperplaneTermType {
        SquaredLinearLossTerm,
        SquaredHingeLossTerm,
    }

    protected final boolean hinge;

    /**
     * Cache the matrices we will use to minimize the terms.
     * Since the matrix (which is a lower triangle) is based on the term's weights and coefficients,
     * there will typically be a lot of redundancy between rules.
     * What we are caching, specifically, is the lower triangle in the Cholesky decomposition of the symmetric matrix:
     * M[i, j] = 2 * weight * coefficients[i] * coefficients[j]
     */
    private static Map<Integer, FloatMatrix> lowerTriangleCache = new HashMap<Integer, FloatMatrix>();

    public static SquaredHyperplaneTerm createSquaredLossTerm(GroundRule groundRule, Hyperplane<LocalVariable> hyperplane) {
        return new SquaredHyperplaneTerm(groundRule, hyperplane, false);
    }

    public static SquaredHyperplaneTerm createSquaredHingeLossTerm(GroundRule groundRule, Hyperplane<LocalVariable> hyperplane) {
        return new SquaredHyperplaneTerm(groundRule, hyperplane, true);
    }

    /**
     * The full constructor is made available, but callers should favor the static creation methods.
     */
    public SquaredHyperplaneTerm(GroundRule groundRule, Hyperplane<LocalVariable> hyperplane, boolean hinge) {
        super(hyperplane, groundRule);

        this.hinge = hinge;
    }

    @Override
    public void minimize(float stepSize, float[] consensusValues) {
        if (getHyperplaneTermType() == HyperplaneTermType.SquaredHingeLossTerm) {
            minimizeSquaredHingeLoss(stepSize, consensusValues);
        } else {
            minimizeSquaredLinearLoss(stepSize, consensusValues);
        }
    }

    @Override
    public float evaluate() {
        if (getHyperplaneTermType() == HyperplaneTermType.SquaredHingeLossTerm) {
            return evaluateSquaredHingeLoss();
        } else {
            return evaluateSquaredLinearLoss();
        }
    }

    @Override
    public float evaluate(float[] consensusValues) {
        if (getHyperplaneTermType() == HyperplaneTermType.SquaredHingeLossTerm) {
            return evaluateSquaredHingeLoss(consensusValues);
        } else {
            return evaluateSquaredLinearLoss(consensusValues);
        }
    }

    public HyperplaneTermType getHyperplaneTermType() {
        if (hinge) {
            return HyperplaneTermType.SquaredHingeLossTerm;
        } else {
            return HyperplaneTermType.SquaredLinearLossTerm;
        }
    }

    // Functionality for SquaredLinearLoss terms.

    public void minimizeSquaredLinearLoss(float stepSize, float[] consensusValues) {
        minWeightedSquaredHyperplane(stepSize, consensusValues);
    }

    /**
     * weight * (coefficients^T * local - constant)^2
     */
    public float evaluateSquaredLinearLoss() {
        return weight * (float)Math.pow(computeInnerPotential(), 2.0);
    }

    /**
     * weight * (coefficients^T * consensus - constant)^2
     */
    public float evaluateSquaredLinearLoss(float[] consensusValues) {
        return weight * (float)Math.pow(computeInnerPotential(consensusValues), 2.0);
    }

    // Functionality for SquaredHingeLoss terms.

    public void minimizeSquaredHingeLoss(float stepSize, float[] consensusValues) {
        // Take a gradient step and see if we are in the flat region.
        float total = 0.0f;
        for (int i = 0; i < size; i++) {
            LocalVariable variable = variables[i];
            variable.setValue(consensusValues[variable.getGlobalId()] - variable.getLagrange() / stepSize);
            total += coefficients[i] * variable.getValue();
        }

        // If we are on the flat region, then we are at a solution.
        if (total <= constant) {
            return;
        }

        // We are in the quadratic region, so solve that to find a solution.
        minWeightedSquaredHyperplane(stepSize, consensusValues);
    }

    /**
     * weight * [max(0, coefficients^T * local - constant)]^2
     */
    public float evaluateSquaredHingeLoss() {
        return weight * (float)Math.pow(Math.max(0.0f, computeInnerPotential()), 2.0);
    }

    /**
     * weight * [max(0, coefficients^T * consensus - constant)]^2
     */
    public float evaluateSquaredHingeLoss(float[] consensusValues) {
        return weight * (float)Math.pow(Math.max(0.0f, computeInnerPotential(consensusValues)), 2.0);
    }

    // General Utilities

    /**
     * Minimizes the term as a weighted, squared hyperplane.
     * This function to minimize takes the form:
     * weight * [coefficients^T * local - constant]^2 + (stepsize / 2) * || local - consensus + lagrange / stepsize ||_2^2.
     *
     * The result of the minimization will be stored in the local variables.
     */
    protected void minWeightedSquaredHyperplane(float stepSize, float[] consensusValues) {
        // Different solving methods will be used depending on the size of the hyperplane.

        // Pre-load the local variable with a term that is common in all the solutions:
        // stepsize * consensus - lagrange + (2 * weight * coefficients * constant).
        for (int i = 0; i < size; i++) {
            float value =
                    stepSize * consensusValues[variables[i].getGlobalId()] - variables[i].getLagrange()
                    + 2.0f * weight * coefficients[i] * constant;

            variables[i].setValue(value);
        }

        // Hyperplanes with only one variable can be solved trivially.
        if (size == 1) {
            LocalVariable variable = variables[0];
            float coefficient = coefficients[0];

            variable.setValue(variable.getValue() / (2.0f * weight * coefficient * coefficient + stepSize));

            return;
        }

        // Hyperplanes with only two variables can be solved fairly easily.
        if (size == 2) {
            LocalVariable variable0 = variables[0];
            LocalVariable variable1 = variables[1];
            float coefficient0 = coefficients[0];
            float coefficient1 = coefficients[1];

            float a0 = 2.0f * weight * coefficient0 * coefficient0 + stepSize;
            float b1 = 2.0f * weight * coefficient1 * coefficient1 + stepSize;
            float a1b0 = 2.0f * weight * coefficient0 * coefficient1;

            variable1.setValue(variable1.getValue() - a1b0 * variable0.getValue() / a0);
            variable1.setValue(variable1.getValue() / (b1 - a1b0 * a1b0 / a0));

            variable0.setValue((variable0.getValue() - a1b0 * variable1.getValue()) / a0);

            return;
        }

        // In the case of larger hyperplanes, we can use a Cholesky decomposition to minimize.

        FloatMatrix lowerTriangle = fetchLowerTriangle(stepSize);

        for (int i = 0; i < size; i++) {
            float newValue = variables[i].getValue();

            for (int j = 0; j < i; j++) {
                newValue -= lowerTriangle.get(i, j) * variables[j].getValue();
            }

            variables[i].setValue(newValue / lowerTriangle.get(i, i));
        }

        for (int i = size - 1; i >= 0; i--) {
            float newValue = variables[i].getValue();

            for (int j = size - 1; j > i; j--) {
                newValue -= lowerTriangle.get(j, i) * variables[j].getValue();
            }

            variables[i].setValue(newValue / lowerTriangle.get(i, i));
        }
    }

    // General Utilities

    /**
     * Get the lower triangle if it already exists, compute and cache it otherwise.
     */
    private FloatMatrix fetchLowerTriangle(float stepSize) {
        int hash = HashCode.build(weight);
        hash = HashCode.build(hash, stepSize);
        for (int i = 0; i < size; i++) {
            hash = HashCode.build(hash, coefficients[i]);
        }

        // First check the cache.
        // Typically, each rule (not ground rule) will have its own lowerTriangle.
        FloatMatrix lowerTriangle = lowerTriangleCache.get(hash);
        if (lowerTriangle != null) {
            return lowerTriangle;
        }

        // If we didn't find it, then synchronize and compute it on this thread.
        return computeLowerTriangle(stepSize, hash);
    }

    /**
     * Actually copute the lower triangle and store it in the cache.
     * There is one triangle per rule, so most ground rules will just pull off the same cache.
     */
    private synchronized FloatMatrix computeLowerTriangle(float stepSize, int hash) {
        // There is still a race condition in the map fetch before getting here,
        // so we will check one more time while synchronized.
        if (lowerTriangleCache.containsKey(hash)) {
            return lowerTriangleCache.get(hash);
        }

        float coefficient = 0.0f;

        FloatMatrix matrix = FloatMatrix.zeroes(size, size);

        for (int i = 0; i < size; i++) {
            // Note that the matrix is symmetric.
            for (int j = i; j < size; j++) {
                if (i == j) {
                    coefficient = 2.0f * weight * coefficients[i] * coefficients[i] + stepSize;
                    matrix.set(i, i, coefficient);
                } else {
                    coefficient = 2.0f * weight * coefficients[i] * coefficients[j];
                    matrix.set(i, j, coefficient);
                    matrix.set(j, i, coefficient);
                }
            }
        }

        matrix.choleskyDecomposition(true);
        lowerTriangleCache.put(hash, matrix);

        return matrix;
    }
}
