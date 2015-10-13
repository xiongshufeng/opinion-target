package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;

import org.apache.commons.io.FileUtils;

import edu.cuhk.hccl.data.Sentences;
import edu.cuhk.hccl.data.Sentences.Sentence;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations.AnswerAnnotation;
import edu.stanford.nlp.ling.CoreLabel;

public class AppOpinionTarget 
{
	
	public static void main(String[] args) throws IOException, JAXBException,
		ClassCastException, ClassNotFoundException {
		
		String crfModelFile = args[0];
		String trainFile = args[1];
		String testFile = args[2];
		String predictFile = args[3];
		
		System.out.println("Processing is started...");
		
		// train crf model
		Properties props = new Properties();
		
		String crfPropFile = "crf-model.prop";
		InputStream propStream = AppOpinionTarget.class.getClassLoader().getResourceAsStream(crfPropFile);
		props.load(propStream);
		props.setProperty("trainFile", trainFile);
		props.setProperty("serializeTo", crfModelFile);
		
		CRFClassifier<CoreLabel> crfClassifier = new CRFClassifier<CoreLabel>(props);
		
		File modelFile = new File(crfModelFile);
		if (modelFile.exists()){
			crfClassifier.loadClassifier(modelFile);
		} else{
			crfClassifier.train();
			crfClassifier.serializeClassifier(crfModelFile);
		}
		
		JAXBContext context = JAXBContext.newInstance(Sentences.class);
		Unmarshaller unmarshaller = context.createUnmarshaller();
		Sentences testSents = (Sentences) unmarshaller.unmarshal(new File(testFile));
		
		List<String> predictResult = new ArrayList<String>();
		
		for (Sentence sent : testSents.getSentence()){
			List<List<CoreLabel>> crfOut = crfClassifier.classify(sent.getText());
			for (List<CoreLabel> sentence : crfOut) {
				for (CoreLabel word : sentence) {
					String wordText = word.originalText().trim();
					if (!wordText.isEmpty()){
						String annotation = word.get(AnswerAnnotation.class);
						predictResult.add(wordText + "\t" + annotation);
					}
				}
			}
			predictResult.add("");
		}
		
		FileUtils.writeLines(new File(predictFile), predictResult, false);

		System.out.println("Processing is finished!");
	}	
}
