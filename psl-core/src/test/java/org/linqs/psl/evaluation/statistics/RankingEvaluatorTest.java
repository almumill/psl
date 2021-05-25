/*
 * This file is part of the PSL software.
 * Copyright 2011-2015 University of Maryland
 * Copyright 2013-2021 The Regents of the University of California
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
package org.linqs.psl.evaluation.statistics;

import static org.junit.Assert.assertEquals;

import org.linqs.psl.application.learning.weight.TrainingMap;
import org.linqs.psl.database.DataStore;
import org.linqs.psl.database.Database;
import org.linqs.psl.database.Partition;
import org.linqs.psl.database.atom.PersistedAtomManager;
import org.linqs.psl.database.loading.Inserter;
import org.linqs.psl.database.rdbms.RDBMSDataStore;
import org.linqs.psl.database.rdbms.driver.H2DatabaseDriver;
import org.linqs.psl.model.atom.GroundAtom;
import org.linqs.psl.model.atom.RandomVariableAtom;
import org.linqs.psl.model.predicate.StandardPredicate;
import org.linqs.psl.model.term.Constant;
import org.linqs.psl.model.term.ConstantType;
import org.linqs.psl.model.term.UniqueIntID;
import org.linqs.psl.util.MathUtils;

import org.junit.Before;
import org.junit.Test;

public class RankingEvaluatorTest extends EvaluatorTest<RankingEvaluator> {
    /**
     * Run several runs against an external implementation
     */
    @Test
    public void testExternalOneCategoryIndex() {
        RankingEvaluator evaluator = new RankingEvaluator(new int[] {1});

        float[] truth = new float[]{0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        int[][] truthArgs = new int[][]{{0, 0}, {0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 2}, {3, 2}, {3, 0}, {4, 0}, {4, 1}};
        float[] predictions = new float[]{0.420f, 0.929f, 0.800f, 0.719f, 0.514f, 0.013f, 0.183f, 0.748f, 0.261f, 0.358f};
        int[][] predictionsArgs = truthArgs;

        init(predictions, truth, predictionsArgs, truthArgs);
        evaluator.compute(trainingMap, predicate);
        assertEquals(0.75f, evaluator.mrr(), MathUtils.EPSILON);

        truth = new float[]{0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};
        truthArgs = new int[][]{{0, 0}, {0, 1}, {1, 2}, {1, 1}, {2, 1}, {2, 0}, {3, 2}, {3, 1}, {4, 2}, {4, 1}};
        predictions = new float[]{0.380f, 0.776f, 0.936f, 0.623f, 0.902f, 0.432f, 0.155f, 0.453f, 0.942f, 0.765f};
        predictionsArgs = truthArgs;

        init(predictions, truth, predictionsArgs, truthArgs);
        evaluator.compute(trainingMap, predicate);
        assertEquals(0.8f, evaluator.mrr(), MathUtils.EPSILON);

        truth = new float[]{0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
        truthArgs = new int[][]{{0, 0}, {0, 2}, {1, 2}, {1, 0}, {2, 1}, {2, 2}, {3, 0}, {3, 1}, {4, 0}, {4, 1}};
        predictions = new float[]{0.372f, 0.179f, 0.208f, 0.562f, 0.470f, 0.346f, 0.754f, 0.564f, 0.638f, 0.448f};
        predictionsArgs = truthArgs;

        init(predictions, truth, predictionsArgs, truthArgs);
        evaluator.compute(trainingMap, predicate);
        assertEquals(0.9f, evaluator.mrr(), MathUtils.EPSILON);
    }

    /**
     * Run several runs with two category indexes against an external implementation.
     */

    @Test
    public void testExternalTwoCategoryIndexes() {
        RankingEvaluator evaluator = new RankingEvaluator(new int[]{1, 2});

        // Tests for the evaluator with multiple category indexes
        float[] truth = new float[]{1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f};
        int[][] truthArgs = new int[][]{{0, 1, 1}, {0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {1, 0, 0}, {2, 0, 0}, {2, 1, 1}, {2, 1, 0}, {3, 0, 0}, {3, 0, 1}, {3, 1, 1}, {4, 0, 1}, {4, 1, 1}, {4, 1, 0}};
        float[] predictions = new float[]{0.333f, 0.751f, 0.420f, 0.645f, 0.143f, 0.237f, 0.474f, 0.991f, 0.701f, 0.703f, 0.616f, 0.227f, 0.722f, 0.467f, 0.405f};
        int[][] predictionsArgs = truthArgs;

        init(predictions, truth, predictionsArgs, truthArgs);
        evaluator.compute(trainingMap, predicate);
        assertEquals(0.583333f, evaluator.mrr(), MathUtils.EPSILON);

        truth = new float[]{0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
        truthArgs = new int[][]{{0, 1, 0}, {0, 0, 1}, {0, 1, 1}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {2, 0, 1}, {2, 1, 0}, {2, 0, 0}, {3, 0, 1}, {3, 0, 0}, {3, 1, 0}, {4, 1, 1}, {4, 0, 1}, {4, 1, 0}};
        predictions = new float[]{0.016f, 0.840f, 0.269f, 0.638f, 0.654f, 0.800f, 0.319f, 0.241f, 0.766f, 0.615f, 0.936f, 0.799f, 0.168f, 0.581f, 0.720f};
        predictionsArgs = truthArgs;

        init(predictions, truth, predictionsArgs, truthArgs);
        evaluator.compute(trainingMap, predicate);
        assertEquals(0.636363f, evaluator.mrr(), MathUtils.EPSILON);

        truth = new float[]{1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
        truthArgs = new int[][]{{0, 1, 1}, {0, 0, 1}, {0, 1, 0}, {1, 1, 1}, {1, 0, 0}, {1, 0, 1}, {2, 0, 0}, {2, 1, 1}, {2, 1, 0}, {3, 0, 1}, {3, 1, 1}, {3, 1, 0}, {4, 0, 0}, {4, 0, 1}, {4, 1, 1}};
        predictions = new float[]{0.898f, 0.816f, 0.389f, 0.812f, 0.302f, 0.312f, 0.471f, 0.771f, 0.085f, 0.047f, 0.480f, 0.249f, 0.188f, 0.410f, 0.912f};
        predictionsArgs = truthArgs;

        init(predictions, truth, predictionsArgs, truthArgs);
        evaluator.compute(trainingMap, predicate);
        assertEquals(0.555555f, evaluator.mrr(), MathUtils.EPSILON);
    }

    /**
     * Test the MRR metric when no entity indexes exist.
     */
    @Test
    public void testNoEntityIndexes() {
        RankingEvaluator evaluator = new RankingEvaluator(new int[] {0,1});

        float[] truth = new float[]{0.0f, 1.0f, 0.0f, 1.0f};
        float[] predictions = new float[]{0.2f, 0.4f, 0.6f, 0.8f};

        init(predictions, truth);
        evaluator.compute(trainingMap, predicate);
        assertEquals(0.666666f, evaluator.mrr(), MathUtils.EPSILON);
    }

    @Override
    protected RankingEvaluator getEvaluator() {
        return new RankingEvaluator();
    }
}
