package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import javax.xml.bind.JAXBException;

import org.apache.commons.io.FileUtils;

import edu.cuhk.hccl.data.Sentences;
import edu.cuhk.hccl.data.Sentences.Sentence;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectCategories;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectTerms;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectCategories.AspectCategory;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectTerms.AspectTerm;
import edu.cuhk.hccl.util.XmlUtil;


public class AppStatistics {
	
	public static void main(String[] args) throws JAXBException{
		
		String lapTrain = "evaluation/Laptop_Train_v2.xml";
		String lapTest = "evaluation/Laptops_Test_Data_PhaseB.xml";
		
		String resTrain = "evaluation/Restaurants_Train_v2.xml";
		String resTest = "evaluation/Restaurants_Test_Data_PhaseB.xml";
		
		printTermsStatistics(lapTrain);
		printTermsStatistics(lapTest);
		
		printTermsStatistics(resTrain);
		printTermsStatistics(resTest);
		
		printPolarityStatistics(lapTrain);
		printPolarityStatistics(resTrain);
		
		printCategoryStatistics(resTrain);
	}

	public static void printTermsStatistics(String fileName) throws JAXBException {
		System.out.println("--------------------------------------");
		System.out.println(fileName);
		System.out.println("--------------------------------------");
		
		Sentences sents = XmlUtil.parseXML(fileName);
		
		// # Sentences
		int numSents = sents.getSentence().size();
		
		// # One-Token Terms
		int numOneToken = 0; 
		
		// # Multi-Token Terms
		int numMultiToken = 0; 
		
		// total length of sentence
		int sentsLength = 0;
		
		// record all aspect terms
		List<String> termsList = new ArrayList<String>();
		
		for (Sentence sent : sents.getSentence()){
			AspectTerms terms = sent.getAspectTerms();
			if (terms != null) {
				for (AspectTerm term : terms.getAspectTerm()) {
					String termValue = term.getTerm();
					String[] words = termValue.split(" ");
					if (words.length == 1){
						numOneToken += 1;
					} else if (words.length > 1){
						numMultiToken += 1;
					}
					termsList.add(term.getTerm());	
				}
			}
			
			sentsLength += sent.getText().split(" ").length;
		}
		
		// # Total Aspect Terms
		int totalTokens = numOneToken + numMultiToken;
		
		System.out.printf("#Sentences \t #One-Token \t #Multi-Token \t #Total Terms\n");
		System.out.printf("%s \t %s \t %s \t %s\n", numSents, numOneToken, numMultiToken, totalTokens);
		if (totalTokens > 0){
			System.out.printf("One-token: %.3f", numOneToken * 100.0 / totalTokens);
			System.out.printf("Multi-token: %.3f", numMultiToken * 100.0 / totalTokens);
		}
		
		System.out.printf("\nAverage words in sentence: %.3f. \n", sentsLength * 1.0 / numSents);
		
		Collections.sort(termsList);
		String termsFile = fileName + ".terms";
		try {
			FileUtils.writeLines(new File(termsFile), termsList, false);
			System.out.printf("All aspect terms are saved at: %s.\n", termsFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void printPolarityStatistics(String fileName) throws JAXBException {
		System.out.println("--------------------------------------");
		System.out.println(fileName);
		System.out.println("--------------------------------------");
		
		Sentences sents = XmlUtil.parseXML(fileName);
		HashMap<String, List<String>> termMap = new HashMap<String, List<String>>();
		HashMap<String, List<String>> categoryMap = new HashMap<String, List<String>>();
		
		for (Sentence sent : sents.getSentence()){
			// Term Polarity
			AspectTerms terms = sent.getAspectTerms();
			if (terms != null) {
				for (AspectTerm term : terms.getAspectTerm()) {
					String key = term.getPolarity();
					if (!termMap.containsKey(key)) {
						termMap.put(key, new ArrayList<String>());
					}
					termMap.get(key).add((sent.getText()));
				}
			}
			
			// Category Polarity
			AspectCategories categories = sent.getAspectCategories();
			if (categories!=null){
				for (AspectCategory category : categories.getAspectCategory()) {
					String key = category.getPolarity();
					if (!categoryMap.containsKey(key)) {
						categoryMap.put(key, new ArrayList<String>());
					}
					categoryMap.get(key).add((sent.getText()));
				}
			}
		}
		
		int numSents = sents.getSentence().size();
		System.out.printf("#Sentences: %d \n", numSents);
		
		int numTerms = 0;
		for (String key : termMap.keySet()){
			numTerms += termMap.get(key).size();
		}
		
		System.out.printf("#Terms: %d \n", numTerms);
		
		int numCategories = 0;
		for (String key : categoryMap.keySet()){
			numCategories += categoryMap.get(key).size();
		}
		
		System.out.printf("#Categories: %d \n", numCategories);
		
		System.out.println("Distributions of aspect terms polarity:");
		for (String key : termMap.keySet()){
			List<String> list = termMap.get(key);
			
			System.out.printf("#%s: %d (%.3f) \n", key, list.size(), list.size() * 100.0 / numTerms);
		}
		
		System.out.println("Distributions of aspect category polarity:");
		for (String key : categoryMap.keySet()){
			List<String> list = categoryMap.get(key);
			
			System.out.printf("#%s: %d (%.3f) \n", key, list.size(), list.size() * 100.0 / numCategories);
		}
	}
	
	public static void printCategoryStatistics(String fileName) throws JAXBException {
		System.out.println("--------------------------------------");
		System.out.println(fileName);
		System.out.println("--------------------------------------");
		
		Sentences sents = XmlUtil.parseXML(fileName);
		HashMap<String, List<String>> categoryMap = new HashMap<String, List<String>>();
		
		for (Sentence sent : sents.getSentence()){			
			// Category Polarity
			AspectCategories categories = sent.getAspectCategories();
			if (categories!=null){
				for (AspectCategory category : categories.getAspectCategory()) {
					String key = category.getCategory();
					if (!categoryMap.containsKey(key)) {
						categoryMap.put(key, new ArrayList<String>());
					}
					categoryMap.get(key).add((sent.getText()));
				}
			}
		}
		
		int numSents = sents.getSentence().size();
		System.out.printf("#Sentences: %d \n", numSents);
		
		int numCategories = 0;
		for (String key : categoryMap.keySet()){
			numCategories += categoryMap.get(key).size();
		}
		
		System.out.printf("#Categories: %d \n", numCategories);
		
		System.out.println("Distributions of aspect categories:");
		for (String key : categoryMap.keySet()){
			List<String> list = categoryMap.get(key);
			
			System.out.printf("#%s: %d (%.3f) \n", key, list.size(), list.size() * 100.0 / numCategories);
		}
	}
}
