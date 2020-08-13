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
import org.linqs.psl.reasoner.term.ReasonerTerm;
import org.linqs.psl.util.MathUtils;

/**
 * Objective term for an ADMMReasoner.
 *
 * TEST(eriq): Fill in description when each class gets brought in.
 * This general class covers three specific types of terms:
 * 1) LinearLoss Terms: weight * coefficients^T * y
 * 2) Hyperplane Terms:
 * 3) Squared Hyperplane Terms:
 * Where y can be either local or concensus values.
 *
 * Minimizing a term comes down to minizing the weighted potential plus a squared norm:
 * weight * [max(0, coefficients^T * local - constant)]^power + (stepsize / 2) * || local - consensus + lagrange / stepsize ||_2^2.
 *
 * The reason these terms are housed in a single class instead of subclasses is for performance
 * in streaming settings where terms must be quickly serialized and deserialized.
 *
 * All coefficients must be non-zero.
 */
public class ADMMObjectiveTerm implements ReasonerTerm {
    /**
     * The specific type of term represented by this instance.
     */
    public static enum TermType {
        LinearLossTerm,
        HyperplaneTerm,
        SquaredHyperplaneTerm
    }

    // TODO(eriq): Remove the reference to a ground rule.
    protected final GroundRule groundRule;

    protected final float weight;
    protected final int size;

    protected final float[] coefficients;
    protected final LocalVariable[] variables;

    protected final float constant;

    // TEST(eriq): Check constructor arg order.

    public static ADMMObjectiveTerm createLinearLossTerm(GroundRule groundRule, Hyperplane<LocalVariable> hyperplane) {
        return new ADMMObjectiveTerm(hyperplane, groundRule);
    }

    /**
     * Construct an ADMM objective term by taking ownership of the hyperplane and all members of it.
     * The full constructor is made available, but callers should favor the static creation methods.
     */
    public ADMMObjectiveTerm(Hyperplane<LocalVariable> hyperplane, GroundRule groundRule) {
        // TEST
        this.groundRule = groundRule;

        this.size = hyperplane.size();

        this.variables = hyperplane.getVariables();
        this.coefficients = hyperplane.getCoefficients();
        this.constant = hyperplane.getConstant();

        if (groundRule instanceof WeightedGroundRule) {
            this.weight = (float)((WeightedGroundRule)groundRule).getWeight();
        } else {
            this.weight = Float.POSITIVE_INFINITY;
        }
    }

    public void updateLagrange(float stepSize, float[] consensusValues) {
        // Use index instead of iterator here so we can see clear results in the profiler.
        // http://psy-lob-saw.blogspot.co.uk/2014/12/the-escape-of-arraylistiterator.html
        for (int i = 0; i < size; i++) {
            LocalVariable variable = variables[i];
            variable.setLagrange(variable.getLagrange() + stepSize * (variable.getValue() - consensusValues[variable.getGlobalId()]));
        }
    }

    /**
     * Get the variables used in this term.
     * The caller should not modify the returned array, and should check size() for a reliable length.
     */
    public LocalVariable[] getVariables() {
        return variables;
    }

    /**
     * Get the number of variables in this term.
     */
    @Override
    public int size() {
        return size;
    }

    public GroundRule getGroundRule() {
        return groundRule;
    }

    /**
     * Get the specific type of term this instance represents.
     */
    public TermType getTermType() {
        // TEST: Simplify when all are here.
        if (!Float.isInfinite(weight) && MathUtils.isZero(constant)) {
            return TermType.LinearLossTerm;
        } else {
            throw new RuntimeException("TEST");
        }
    }

    /**
     * Modify the local variables to minimize this term (within the bounds of the step size).
     */
    public void minimize(float stepSize, float[] consensusValues) {
        TermType termType = getTermType();
        if (termType == TermType.LinearLossTerm) {
            minimizeLinearLoss(stepSize, consensusValues);
        } else {
            throw new RuntimeException("TEST");
        }
    }

    /**
     * Evaluate this potential using the local variables.
     */
    public float evaluate() {
        TermType termType = getTermType();
        if (termType == TermType.LinearLossTerm) {
            return evaluateLinearLoss();
        } else {
            throw new RuntimeException("TEST");
        }
    }

    /**
     * Evaluate this potential using the given consensus values.
     */
    public float evaluate(float[] consensusValues) {
        if (getTermType() == TermType.LinearLossTerm) {
            return evaluateLinearLoss(consensusValues);
        } else {
            throw new RuntimeException("TEST");
        }
    }

    // Functionality for linear loss terms.

    private void minimizeLinearLoss(float stepSize, float[] consensusValues) {
        // Linear losses can be directly minimized.

        for (int i = 0; i < size; i++) {
            LocalVariable variable = variables[i];

            float value =
                    consensusValues[variable.getGlobalId()]
                    - variable.getLagrange() / stepSize
                    - (weight * coefficients[i] / stepSize);

            variable.setValue(value);
        }
    }

    /**
     * weight * coefficients^T * local
     */
    private float evaluateLinearLoss() {
        return weight * computeInnerPotential();
    }

    /**
     * weight * coefficients^T * consensus
     */
    private float evaluateLinearLoss(float[] consensusValues) {
        return weight * computeInnerPotential(consensusValues);
    }




    // General Utilities

    /**
     * coefficients^T * local - constant
     */
    protected float computeInnerPotential() {
        float value = 0.0f;
        for (int i = 0; i < size; i++) {
            value += coefficients[i] * variables[i].getValue();
        }

        return value - constant;
    }

    /**
     * coefficients^T * consensus - constant
     */
    protected float computeInnerPotential(float[] consensusValues) {
        float value = 0.0f;
        for (int i = 0; i < size; i++) {
            value += coefficients[i] * consensusValues[variables[i].getGlobalId()];
        }

        return value - constant;
    }
}
