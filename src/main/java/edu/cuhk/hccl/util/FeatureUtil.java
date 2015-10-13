package edu.cuhk.hccl.util;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

public class FeatureUtil {
	
	public static void featureIDF(InstanceList instances) {
		Alphabet alphabet = instances.getPipe().getDataAlphabet();
		Object[] tokens = alphabet.toArray();
		
		// determine document frequency for each term
		int[] df = new int[tokens.length];
		for (Instance instance : instances) {
			FeatureVector fv = (FeatureVector) instance.getData();
			int[] indices = fv.getIndices();
			for (int index: indices) {
				df[index]++;
			}
		}
		
		// idf term weighting
		int N = instances.size();
		for (Instance instance : instances) {
			FeatureVector fv = (FeatureVector) instance.getData();
			int[] indices = fv.getIndices();
			for (int index: indices) {
				double tf = fv.value(index);
				double idfcomp = Math.log((double)N/(double)df[index]) / Math.log(N+1);
				fv.setValue(index, tf * idfcomp);
			}
		}
	}
	
	public static void featureTFIDF(InstanceList instances) {
		Alphabet alphabet = instances.getPipe().getDataAlphabet();
		Object[] tokens = alphabet.toArray();
		
		// determine document frequency for each term
		int[] df = new int[tokens.length];
		for (Instance instance : instances) {
			FeatureVector fv = (FeatureVector) instance.getData();
			int[] indices = fv.getIndices();
			for (int index: indices) {
				df[index]++;
			}
		}
		
		// tfidf term weighting
		int N = instances.size();

		// determine document length for each document
		int[] lend = new int[N];
		double lenavg = 0;
		for (int i = 0; i < N; i++) {
			Instance instance = instances.get(i);
			FeatureVector fv = (FeatureVector) instance.getData();
			int[] indices = fv.getIndices();
			double length = 0.0;
			for (int index : indices) {
				length += fv.value(index);
			}
			lend[i] = (int) length;
			lenavg += length;
		}
		if (N > 1) {
			lenavg /= (double) N;
		}

		for (int i = 0; i < N; i++) {
			Instance instance = instances.get(i);
			FeatureVector fv = (FeatureVector) instance.getData();
			int[] indices = fv.getIndices();
			for (int index : indices) {
				double tf = fv.value(index);
				double tfcomp = tf
						/ (tf + 0.5 + 1.5 * (double) lend[i] / lenavg);
				double idfcomp = Math.log((double) N / (double) df[index])
						/ Math.log(N + 1);
				fv.setValue(index, tfcomp * idfcomp);
			}
		}
	}
}
