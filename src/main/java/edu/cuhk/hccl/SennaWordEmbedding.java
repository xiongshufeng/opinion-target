package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;

public class SennaWordEmbedding extends WordEmbedding {

	public final static String WORDS_FILE = "words.lst";
	public final static String EMBEDDING_FILE = "embeddings.txt";

	public SennaWordEmbedding(String modelPath) {
		super(modelPath);
	}

	@Override
	public void loadWordVectors() throws IOException {
		List<String> words = FileUtils.readLines(new File(modelPath + "/" + WORDS_FILE));
		List<String> embeddings = FileUtils.readLines(new File(modelPath + "/" + EMBEDDING_FILE));

		for (int i = 0; i < words.size(); i++) {
			String word = words.get(i);
			String[] embArray = embeddings.get(i).split(Utility.SPACE);
			float[] embedding = new float[embArray.length];

			for (int j = 0; j < embArray.length; j++)
				embedding[j] = Float.parseFloat(embArray[j]);

			wordVecMap.put(word, embedding);
		}

	}
}