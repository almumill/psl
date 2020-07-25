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
import org.linqs.psl.reasoner.term.Hyperplane;

/**
 * Objective term for an ADMMReasoner that is based on a hyperplane in some way.
 *
 * Stores the characterization of the hyperplane as coefficients^T * x = constant
 * and projects onto the hyperplane.
 *
 * All coefficients must be non-zero.
 */
public abstract class HyperplaneTerm extends ADMMObjectiveTerm {
    protected final float[] coefficients;
    protected final float constant;

    // These variables are used when solving the objective function.
    // We keep them as member data to avoid multiple allocations.

    /**
     * The optimizer considering only the consensus values (and not the constraint imposed by this local hyperplane).
     * This optimizer will be projected onto this hyperplane to minimize.
     */
    protected final float[] consensusOptimizer;
    protected final float[] unitNormal;

    public HyperplaneTerm(GroundRule groundRule, Hyperplane<LocalVariable> hyperplane) {
        super(hyperplane, groundRule);

        coefficients = hyperplane.getCoefficients();
        constant = hyperplane.getConstant();

        if (size == 1) {
            // If the hyperplane only has one random variable, we can take shortcuts solving it.
            consensusOptimizer = null;
            unitNormal = null;
        } else {
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
    }

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

    /**
     * coefficients^T * x - constant
     */
    @Override
    public float evaluate() {
        float value = 0.0f;
        for (int i = 0; i < size; i++) {
            value += coefficients[i] * variables[i].getValue();
        }

        return value - constant;
    }

    @Override
    public float evaluate(float[] consensusValues) {
        float value = 0.0f;
        for (int i = 0; i < size; i++) {
            value += coefficients[i] * consensusValues[variables[i].getGlobalId()];
        }

        return value - constant;
    }
}
