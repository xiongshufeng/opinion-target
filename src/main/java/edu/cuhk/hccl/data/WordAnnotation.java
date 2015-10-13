package edu.cuhk.hccl.data;

import java.util.Map;
import java.util.TreeMap;

public class WordAnnotation {
	
	private String word;
	private String annotation;
	private String[] annotations;
	
	public static final String[] REST_ENTITIES= new String[]
			{"FOOD", "DRINKS", "SERVICE", "AMBIENCE", "LOCATION", "RESTAURANT"};
	public static final String[] REST_ATTRIBUTES= new String[]
			{"GENERAL", "PRICES", "QUALITY", "STYLE&OPTIONS", "MISCELLANEOUS"};
	
	public static final String[] TARGETS = new String[]{"TERM"};
	public static final String[] POLARITIES = new String[]{"POSITIVE", "NEGATIVE", "NEUTRAL"};
	
	public static final String OTHER = "O";
	
	public WordAnnotation(String[] annotations){
		this.setAnnotations(annotations);
	}
	
	public WordAnnotation(String word, String annotation) {
		this.setWord(word);
		this.setAnnotation(annotation);
	}

	public String getWord() {
		return word;
	}

	public void setWord(String word) {
		this.word = word;
	}

	public String getAnnotation() {
		return annotation;
	}

	public void setAnnotation(String annotation) {
		this.annotation = annotation;
	}

	public String[] getAnnotations() {
		return annotations;
	}

	public void setAnnotations(String[] annotations) {
		this.annotations = annotations;
	}

	public static String[] createAnnotations(String type) {
		
		String[] tmpAnnos = null;

		if (type.equalsIgnoreCase("target"))
			tmpAnnos = TARGETS;
		else if (type.equalsIgnoreCase("polarity"))
			tmpAnnos = POLARITIES;
		else if (type.equalsIgnoreCase("restEntity"))
			tmpAnnos = REST_ENTITIES;
		else if (type.equalsIgnoreCase("restAttribute"))
			tmpAnnos = REST_ATTRIBUTES;

		String[] annos = new String[2 * tmpAnnos.length + 1];

		for (int i = 0; i < tmpAnnos.length; i++) {
			annos[2 * i] = labelToAnnotation(tmpAnnos[i], true);
			annos[2 * i + 1] = labelToAnnotation(tmpAnnos[i], false);
		}

		annos[2 * tmpAnnos.length] = OTHER;

		return annos;
	}

	public int annotationToNumber(String annotation) {
		
		int number = annotations.length - 1; // Default annotation is OTHER(O)
		for (int i = 0; i < annotations.length; i++){
			if (annotations[i].startsWith(annotation)){
				number = i;
				break;
			}
		}
		
		return number;
	}
	
	public static String labelToAnnotation(String label, boolean isBegin) {
		
		String prefix = isBegin ? "B-" : "I-";
		
		return prefix + label.toUpperCase();
	}
	
	public Map<String, Integer> createLabelIndexMap() {

		Map<String, Integer> label2idx = new TreeMap<String, Integer>();

		for (int i = 0; i < annotations.length; i++) {
			label2idx.put(annotations[i], i);
		}

		return label2idx;
	}
	
}
