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

import org.linqs.psl.application.learning.weight.TrainingMap;
import org.linqs.psl.config.Options;
import org.linqs.psl.model.atom.GroundAtom;
import org.linqs.psl.model.atom.ObservedAtom;
import org.linqs.psl.model.atom.RandomVariableAtom;
import org.linqs.psl.model.predicate.Predicate;
import org.linqs.psl.model.predicate.StandardPredicate;
import org.linqs.psl.model.term.Constant;
import org.linqs.psl.util.StringUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Compute ranking-based statistics.
 */
public class RankingEvaluator extends Evaluator {
    public enum RepresentativeMetric {
        MRR
    }

    public static final String DELIM = ":";

    private RepresentativeMetric representative;
    private String defaultPredicate;
    private List<Integer> categoryIndexes;

    // This is the truth value threshold that truth atoms must pass
    // for us to use their corresponding target atom in a ranking
    // metric calculation.
    private double threshold;

    // Save the map from target atoms to truth atoms
    // for truth value lookups in ranking metric calculations.
    private Map<RandomVariableAtom, ObservedAtom> labelMap;

     // Maintain a map from predicted atoms
     // to their ranks (with respect to their individual
     // entities) for use in ranking metric calculations.
    private Map<GroundAtom, Integer> rankMap;

    public RankingEvaluator() {
        this(RepresentativeMetric.valueOf(Options.EVAL_RANK_REPRESENTATIVE.getString()),
                StringUtils.splitInt(Options.EVAL_RANK_CATEGORY_INDEXES.getString(), DELIM));
    }

    public RankingEvaluator(int... rawCategoryIndexes) {
        this(Options.EVAL_RANK_REPRESENTATIVE.getString(), rawCategoryIndexes);
    }

    public RankingEvaluator(String representative, int... rawCategoryIndexes) {
        this(RepresentativeMetric.valueOf(representative.toUpperCase()), rawCategoryIndexes);
    }

    public RankingEvaluator(RepresentativeMetric representative, int... rawCategoryIndexes) {
        this.representative = representative;

        categoryIndexes = new ArrayList<Integer>(rawCategoryIndexes.length);

        for (int catIndex : rawCategoryIndexes) {
            categoryIndexes.add(catIndex);
        }

        threshold = Options.EVAL_RANK_THRESHOLD.getDouble();
        defaultPredicate = Options.EVAL_RANK_DEFAULT_PREDICATE.getString();
    }

    /**
     * Grab the Constant arguments of an atom at specified positions.
     * We make the list of Constants immutable because entities are immutable
     */
    private List<Constant> getArgsAtPositions(GroundAtom atom, List<Integer> indexes) {
        Constant[] atomArgs = atom.getArguments();
        List<Constant> argsAtPositions = new ArrayList<Constant>(indexes.size());

        for (int index : indexes) {
            argsAtPositions.add(atomArgs[index]);
        }

        return Collections.unmodifiableList(argsAtPositions);
    }

    @Override
    public void compute(TrainingMap trainingMap) {
        if (defaultPredicate == null) {
            throw new UnsupportedOperationException("RankingEvaluators must have a default predicate set (through config).");
        }

        compute(trainingMap, StandardPredicate.get(defaultPredicate));
    }

    @Override
    public void compute(TrainingMap trainingMap, StandardPredicate predicate) {
        labelMap = trainingMap.getLabelMap();
        rankMap = new HashMap<GroundAtom, Integer>();

        int predicateArity = predicate.getArity();
        List<Integer> entityIndexes = new ArrayList<Integer>(predicateArity - categoryIndexes.size());

        for (int index = 0; index < predicateArity; index++) {
            if (categoryIndexes.contains(index)) {
                continue;
            } else {
                entityIndexes.add(index);
            }
        }

        // We keep a map from entities to ArrayLists of all GroundAtoms which contain that entity.
        Map<List<Constant>, List<GroundAtom>> sortedAtoms = new HashMap<List<Constant>, List<GroundAtom>>();

        // Partition the predicted atoms into Lists
        // according to what entity they contain.
        for (GroundAtom atom : trainingMap.getAllPredictions()) {
            if (atom.getPredicate() != predicate) {
                continue;
            }

            List<Constant> entity = getArgsAtPositions(atom, entityIndexes);

            if (!sortedAtoms.containsKey(entity)) {
                sortedAtoms.put(entity, new ArrayList<GroundAtom>());
            }

            sortedAtoms.get(entity).add(atom);
        }

        // Sort these Lists and populate the rankMap.
        for (ArrayList<Constant> entity : sortedAtoms.keySet()) {
            Collections.sort(sortedAtoms.get(entity));
            for (GroundAtom atom : sortedAtoms.get(entity)) {
                rankMap.put(atom, sortedAtoms.get(entity).indexOf(atom) + 1);
            }
        }
    }

    /**
     * Returns the mean reciprocal rank of target atoms
     * that have a corresponding truth atom with a value above
     * the specified threshold.
     */
    public double mrr() {
        // These are the numerator and denominator of the MRR.
        // recRankSum keeps a running sum of reciprocal ranks,
        // and rankedAtomCount keeps track of how many
        // atoms are being ranked in this evaluation.
        double recRankSum = 0.0;
        int rankedAtomCount = 0;

        for (Map.Entry<GroundAtom, Integer> entry : rankMap.entrySet()) {
            // If the target atom's corresponding truth value is below
            // the threshold or it doesn't have a match truth atom, we skip it.
            if (!labelMap.containsKey(entry.getKey()) || labelMap.get(entry.getKey()).getValue() < threshold) {
                continue;
            }

            recRankSum += 1.0 / (double)entry.getValue();
            rankedAtomCount++;
        }

        return recRankSum / (double)rankedAtomCount;
    }

    @Override
    public double getRepMetric() {
        switch (representative) {
            case MRR:
                return mrr();
            default:
                throw new IllegalStateException("Unknown representative metric: " + representative);
        }
    }

    @Override
    public double getBestRepScore() {
        switch (representative) {
            case MRR:
                return 1.0;
            default:
                throw new IllegalStateException("Unknown representative metric: " + representative);
        }
    }

    @Override
    public boolean isHigherRepBetter() {
        return true;
    }

    @Override
    public String getAllStats() {
        return String.format("MRR: %f", mrr());
    }
}
