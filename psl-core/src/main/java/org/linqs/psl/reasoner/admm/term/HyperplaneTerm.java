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
import org.linqs.psl.reasoner.function.FunctionComparator;
import org.linqs.psl.reasoner.term.Hyperplane;
import org.linqs.psl.util.MathUtils;

/**
 * Objective term for an ADMMReasoner that is based on a hyperplane in some way.
 *
 * This general class covers two specific types of terms:
 * 1) HingeLossTerm: weight * max(0, coefficients^T * y - constant).
 * 2) LinearConstraintTerm: (0 if coefficients^T * y [comparator] constant) or (infinity otherwise)
 * Where y can be either local or concensus values.
 *
 * The reason these terms are housed in a single class instead of subclasses is for performance
 * in streaming settings where terms must be quickly serialized and deserialized.
 *
 * All coefficients must be non-zero.
 */
public class HyperplaneTerm extends ADMMObjectiveTerm {
    /**
     * The specific type of term represented by this instance.
     */
    public static enum HyperplaneTermType {
        HingeLossTerm,
        LinearConstraintTerm
    }

    /**
     * When non-null, this term must be a hard constraint.
     */
    protected final FunctionComparator comparator;

    // These variables are used when solving the objective function.
    // We keep them as member data to avoid multiple allocations.
    // However, they may be null when they don't apply to the specific type of term.

    /**
     * The optimizer considering only the consensus values (and not the constraint imposed by this local hyperplane).
     * This optimizer will be projected onto this hyperplane to minimize.
     */
    protected final float[] consensusOptimizer;
    protected final float[] unitNormal;

    public static HyperplaneTerm createHingeLossTerm(GroundRule groundRule, Hyperplane<LocalVariable> hyperplane) {
        return new HyperplaneTerm(groundRule, hyperplane, null);
    }

    public static HyperplaneTerm createLinearConstraintTerm(GroundRule groundRule, Hyperplane<LocalVariable> hyperplane, FunctionComparator comparator) {
        return new HyperplaneTerm(groundRule, hyperplane, comparator);
    }

    /**
     * The full constructor is made available, but callers should favor the static creation methods.
     */
    public HyperplaneTerm(GroundRule groundRule, Hyperplane<LocalVariable> hyperplane, FunctionComparator comparator) {
        super(hyperplane, groundRule);

        this.comparator = comparator;

        // If the hyperplane only has one random variable, we can take shortcuts solving it.
        if (size == 1) {
            consensusOptimizer = null;
            unitNormal = null;
            return;
        }

        consensusOptimizer = new float[size];
        unitNormal = new float[size];

        float length = 0.0f;
        for (int i = 0; i < size; i++) {
            length += coefficients[i] * coefficients[i];
        }
        length = (float)Math.sqrt(length);

        for (int i = 0; i < size; i++) {
            unitNormal[i] = coefficients[i] / length;
        }
    }

    @Override
    public void minimize(float stepSize, float[] consensusValues) {
        if (getHyperplaneTermType() == HyperplaneTermType.HingeLossTerm) {
            minimizeHingeLoss(stepSize, consensusValues);
        } else {
            minimizeConstraint(stepSize, consensusValues);
        }
    }

    @Override
    public float evaluate() {
        if (getHyperplaneTermType() == HyperplaneTermType.HingeLossTerm) {
            return evaluateHingeLoss();
        } else {
            return evaluateConstraint();
        }
    }

    @Override
    public float evaluate(float[] consensusValues) {
        if (getHyperplaneTermType() == HyperplaneTermType.HingeLossTerm) {
            return evaluateHingeLoss(consensusValues);
        } else {
            return evaluateConstraint(consensusValues);
        }
    }

    public HyperplaneTermType getHyperplaneTermType() {
        if (comparator == null) {
            return HyperplaneTermType.HingeLossTerm;
        } else {
            return HyperplaneTermType.LinearConstraintTerm;
        }
    }

    // Functionality for hinge-loss terms.

    public void minimizeHingeLoss(float stepSize, float[] consensusValues) {
        // Look to see if the solution is in one of three sections (in increasing order of difficulty):
        // 1) The flat region.
        // 2) The linear region.
        // 3) The hinge point.

        // Take a gradient step and see if we are in the flat region.
        float total = 0.0f;
        for (int i = 0; i < size; i++) {
            LocalVariable variable = variables[i];
            variable.setValue(consensusValues[variable.getGlobalId()] - variable.getLagrange() / stepSize);
            total += (coefficients[i] * variable.getValue());
        }

        // If we are on the flat region, then we are at a solution.
        if (total <= constant) {
            return;
        }

        // Take a gradient step and see if we are in the linear region.
        total = 0.0f;
        for (int i = 0; i < size; i++) {
            LocalVariable variable = variables[i];
            variable.setValue((consensusValues[variable.getGlobalId()] - variable.getLagrange() / stepSize) - (weight * coefficients[i] / stepSize));
            total += coefficients[i] * variable.getValue();
        }

        // If we are in the linear region, then we are at a solution.
        if (total >= constant) {
            return;
        }

        // We are on the hinge, project to find the solution.
        project(stepSize, consensusValues);
    }

    /**
     * weight * max(0.0, coefficients^T * local - constant)
     */
    public float evaluateHingeLoss() {
        return weight * Math.max(0.0f, computeInnerPotential());
    }

    /**
     * weight * max(0.0, coefficients^T * consensus - constant)
     */
    public float evaluateHingeLoss(float[] consensusValues) {
        return weight * Math.max(0.0f, computeInnerPotential(consensusValues));
    }

    // Functionality for constraint terms.

    public float evaluateConstraint() {
        return evaluateConstraint(null);
    }

    /**
     * Evalauate to zero if the constraint is satisfied, infinity otherwise.
     * if (coefficients^T * y [comparator] constant) { return 0.0 }
     * else { return infinity }
     */
    private float evaluateConstraint(float[] consensusValues) {
        float value = 0.0f;
        if (consensusValues == null) {
            value = computeInnerPotential();
        } else {
            value = computeInnerPotential(consensusValues);
        }

        if (comparator.equals(FunctionComparator.EQ)) {
            if (MathUtils.isZero(value, MathUtils.RELAXED_EPSILON)) {
                return 0.0f;
            }
            return Float.POSITIVE_INFINITY;
        } else if (comparator.equals(FunctionComparator.LTE)) {
            if (value <= 0.0f) {
                return 0.0f;
            }
            return Float.POSITIVE_INFINITY;
        } else if (comparator.equals(FunctionComparator.GTE)) {
            if (value >= 0.0f) {
                return 0.0f;
            }
            return Float.POSITIVE_INFINITY;
        } else {
            throw new IllegalStateException("Unknown comparison function.");
        }
    }

    public void minimizeConstraint(float stepSize, float[] consensusValues) {
        // If the constraint is an inequality, then we may be able to solve without projection.
        if (!comparator.equals(FunctionComparator.EQ)) {
            float total = 0.0f;

            // Take the lagrange step and see if that is the solution.
            for (int i = 0; i < size; i++) {
                LocalVariable variable = variables[i];
                variable.setValue(consensusValues[variable.getGlobalId()] - variable.getLagrange() / stepSize);
                total += coefficients[i] * variable.getValue();
            }

            // If the constraint is satisfied, them we are done.
            if ((comparator.equals(FunctionComparator.LTE) && total <= constant)
                    || (comparator.equals(FunctionComparator.GTE) && total >= constant)) {
                return;
            }
        }

        // If the naive minimization didn't work, or if it's an equality constraint,
        // then project onto the hyperplane.
        project(stepSize, consensusValues);
    }

    // General Utilities

    /**
     * Project the solution to the consensus problem onto this hyperplane,
     * thereby finding the min solution.
     * The consensus problem is:
     * [argmin_local stepSize / 2 * ||local - consensus + lagrange / stepSize ||_2^2],
     * while this hyperplane is: [coefficients^T * local = constant].
     * The result of the projection is stored in the local variables.
     */
    protected void project(float stepSize, float[] consensusValues) {
        // When there is only one variable, there is only one answer.
        // This answer must satisfy the constraint.
        if (size == 1) {
            variables[0].setValue(constant / coefficients[0]);
            return;
        }

        // ConsensusOptimizer = Projection + (multiplier)(unitNormal).
        // Note that the projection is in this hyperplane and therefore orthogonal to the unitNormal.
        // So, these two orthogonal components can makeup the consensusOptimizer.

        // Get the min w.r.t. to the consensus values.
        // This is done by taking a step according to the lagrange.
        for (int i = 0; i < size; i++) {
            consensusOptimizer[i] = consensusValues[variables[i].getGlobalId()] - variables[i].getLagrange() / stepSize;
        }

        // Get the length of the normal.
        // Any matching index can be used to compute the length.
        float length = coefficients[0] / unitNormal[0];

        // Get the multiplier to the unit normal that properly scales it to match the consensus optimizer.
        // We start with the constant, because it is actually part of our vector,
        // but since it always has a 1 cofficient we treat it differently.
        float multiplier = -1.0f * constant / length;
        for (int i = 0; i < size; i++) {
            multiplier += consensusOptimizer[i] * unitNormal[i];
        }

        // Projection = ConsensusOptimizer - (multiplier)(unitNormal).
        for (int i = 0; i < size; i++) {
            variables[i].setValue(consensusOptimizer[i] - multiplier * unitNormal[i]);
        }
    }
}
