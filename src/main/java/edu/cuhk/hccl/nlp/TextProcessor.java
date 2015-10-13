package edu.cuhk.hccl.nlp;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import edu.cuhk.hccl.AppPrepareRnnDataset;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class TextProcessor {

	public static final Set<String> JJ_TAGS = new HashSet<String>(Arrays.asList(new String[] { "JJ", "JJR", "JJS" }));
	public static final Set<String> NN_TAGS = new HashSet<String>(Arrays.asList(new String[] { "NN", "NNS", "NNP", "NNPS"}));
	public static final Set<String> RB_TAGS = new HashSet<String>(Arrays.asList(new String[] { "RB", "RBR", "RBS" }));
	public static final Set<String> VB_TAGS = new HashSet<String>(Arrays.asList(new String[] { "VB", "VBD", "VBG" , "VBN", "VBP", "VBZ"}));
	
	public static final List<String> FEATURES = Arrays.asList(
			new String[]{"JJ", "NN", "RB", "VB",
					"B-NP", "B-PP", "B-VP", "B-ADJP", "B-ADVP",
					"I-NP", "I-PP", "I-VP", "I-ADJP", "I-ADVP"});
	
	public enum Type{Tagger, Parser};
	
	private StanfordCoreNLP coreNLP = null;
	
	protected TextProcessor(){
		
	}
	
	public TextProcessor(Properties props){
		this.coreNLP = new StanfordCoreNLP(props);
	}
	
	public static Properties getProperties(Type type){
		Properties props = new Properties();
		
		switch(type){
		case Tagger:
			props.put("annotators", "tokenize, ssplit, pos, lemma");
		    props.put("pos.model", "taggers/english-left3words-distsim.tagger");
			break;
		case Parser:
			props.setProperty("annotators", "tokenize, ssplit, parse");
			props.setProperty("parse.model", "parsers/englishPCFG.ser.gz");
			break;
		default:
			props.put("annotators", "tokenize, ssplit, pos, lemma");
		    props.put("pos.model", "taggers/english-left3words-distsim.tagger");
			break;
		}
		
		return props;
	}

	public StanfordCoreNLP getCoreNLP() {
		return coreNLP;
	}
	
	public String extractTokens(String text, Set<String> tags) {
        Annotation document = new Annotation(text);
        coreNLP.annotate(document);
        
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        StringBuffer tokens = new StringBuffer();
        
        for (CoreMap sentence : sentences) {
            List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
            for (CoreLabel label : labels) {
                if (tags.contains(label.get(PartOfSpeechAnnotation.class))) {
                	tokens.append(label.lemma() + " ");
                }
            }
        }
        
        return tokens.toString();
    }
	
	public static ChunkerME loadChunker() {
		InputStream modelIn = null;
		ChunkerModel model = null;

		try {
			ClassLoader classLoader = AppPrepareRnnDataset.class.getClassLoader();
			File file = new File(classLoader.getResource("chunker/en-chunker.bin").getFile());
			modelIn = new FileInputStream(file);
			model = new ChunkerModel(modelIn);
		} catch (IOException e) {
			// Model loading failed, handle the error
			e.printStackTrace();
		} finally {
			if (modelIn != null) {
				try {
					modelIn.close();
				} catch (IOException e) {
				}
			}
		}

		ChunkerME chunker = new ChunkerME(model);
		return chunker;
	}

}
