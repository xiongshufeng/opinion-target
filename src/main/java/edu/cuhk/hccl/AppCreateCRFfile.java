package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;

import org.apache.commons.io.FileUtils;

import edu.cuhk.hccl.data.Sentences;
import edu.cuhk.hccl.data.Sentences.Sentence;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectTerms;
import edu.cuhk.hccl.data.Sentences.Sentence.AspectTerms.AspectTerm;
import edu.cuhk.hccl.data.WordAnnotation;
import edu.cuhk.hccl.nlp.Parser;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class AppCreateCRFfile {
	
	public static final String SEPERATOR = "\t";
	
	public static void main(String[] args) throws IOException, JAXBException {

		System.out.println("Processing is started...");

		JAXBContext context = JAXBContext.newInstance(Sentences.class);
		Unmarshaller unmarshaller = context.createUnmarshaller();

		Sentences sents = (Sentences) unmarshaller.unmarshal(new File(args[0]));

		createCRFfile(args[1], sents);

		System.out.println("Processing is finished!");
	}

	public static void createCRFfile(String crfFile, Sentences sents)
			throws IOException {
		
		List<String> lines = new ArrayList<String>();
		StanfordCoreNLP tagger = Parser.createTagger();
		
		for (Sentence sent : sents.getSentence()) {
			
			AspectTerms terms = sent.getAspectTerms();
			Queue<WordAnnotation> termsQueue = new LinkedList<WordAnnotation>();
			
			if (terms != null){
				List<AspectTerm> aspects = sent.getAspectTerms().getAspectTerm();
				for(AspectTerm term : aspects){
					String[] words = term.getTerm().split(" ");
					termsQueue.add(new WordAnnotation(words[0],WordAnnotation.labelToAnnotation("TERM", true)));
					
					for (int i = 1; i < words.length; i++){
						termsQueue.add(new WordAnnotation(words[i], WordAnnotation.labelToAnnotation("TERM", false)));
					}
				}
			}
			
			Annotation annotation = tagger.process(sent.getText());
			for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
				List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
				for (CoreLabel label : labels){
					WordAnnotation peek = termsQueue.peek();
					String originalText = label.originalText();
					
					if (!originalText.isEmpty()) {
						if (peek != null && originalText.equalsIgnoreCase(peek.getWord())) {
							lines.add(originalText + SEPERATOR + label.tag() + SEPERATOR
									+ peek.getAnnotation());
							termsQueue.remove();
						} else {
							lines.add(originalText + SEPERATOR + label.tag() + SEPERATOR
									+ WordAnnotation.OTHER);
						}
					}
				}
			}
			
			lines.add("");
		}

		FileUtils.writeLines(new File(crfFile), lines, false);
	}
}