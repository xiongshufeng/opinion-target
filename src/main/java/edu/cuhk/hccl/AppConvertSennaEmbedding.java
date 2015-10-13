package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.xml.bind.JAXBException;

import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FileUtils;

public class AppConvertSennaEmbedding {

	public static void main(String[] args) throws JAXBException, IOException, ParseException {

		System.out.println("[INFO] Processing is started...");

		String modelPath = args[0];
		File outFile = new File(args[1]);
		
		List<String> words = FileUtils.readLines(
				new File(modelPath + "/" + SennaWordEmbedding.WORDS_FILE));
		
		List<String> embeddings = FileUtils.readLines(
				new File(modelPath + "/" + SennaWordEmbedding.EMBEDDING_FILE));

		FileUtils.write(outFile, embeddings.size() + " 50" + "\n", false);
		
		for (int i = 0; i < words.size(); i++) {
			String line = words.get(i) + Utility.SPACE + embeddings.get(i) + "\n";
			FileUtils.write(outFile, line, true);
		}

		
		System.out.println("[INFO] Result is written to: " + outFile);
		System.out.println("[INFO] Processing is finished!");
	}

	
}
