package edu.cuhk.hccl.nlp;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;

import edu.cuhk.hccl.data.Aspect;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.CoreMap;

public class Parser {
	
	public static StanfordCoreNLP createParser(){
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, parse");
		props.setProperty("parse.model", "parsers/englishPCFG.ser.gz");
	 
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    return pipeline;
	}
	
	public static StanfordCoreNLP createTagger(){
		Properties props = new Properties();
		
		props.put("annotators", "tokenize, ssplit, pos, lemma");
	    props.put("pos.model", "taggers/english-left3words-distsim.tagger");
	        
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    return pipeline;
	}
	
	public static List<Aspect> extractAspects(StanfordCoreNLP parser, String text) {
		Annotation annotation = parser.process(text);

		List<Aspect> aspects = new ArrayList<Aspect>();
		for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
			TreebankLanguagePack tlp = new PennTreebankLanguagePack();
			GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
			GrammaticalStructure gs = gsf.newGrammaticalStructure(tree);
			Collection<TypedDependency> td = gs.typedDependenciesCollapsed();

			Object[] objs = td.toArray();
			
			// Handle 1-4 patterns
			TypedDependency acomp = null, cop=null, dobj=null, nsubj=null, neg=null;
			for (Object obj : objs) {
				TypedDependency dependency = (TypedDependency) obj;
				String term = dependency.gov().nodeString();
				String modifier = dependency.dep().nodeString();
				
				if (dependency.reln().getShortName().equals("amod")) {	
					aspects.add(new Aspect(term, modifier));
				} else if (dependency.reln().getShortName().equals("acomp")) {
					acomp = dependency;
				} else if (dependency.reln().getShortName().equals("cop")) {
					cop = dependency;
				} else if (dependency.reln().getShortName().equals("dobj")) {
					dobj = dependency;
				} else if (dependency.reln().getShortName().equals("nsubj")) {
					nsubj = dependency;
				} else if (dependency.reln().getShortName().equals("neg")) {
					neg = dependency;
				}
			}
			
			if (nsubj != null){
				String term = nsubj.gov().nodeString();
				String modifier = nsubj.dep().nodeString();
				if(acomp != null){
					if (term.equals(acomp.gov().nodeString())){
						Aspect aspect = new Aspect(modifier, acomp.dep().nodeString());
						aspects.add(aspect);
					}
				}
				if(cop != null){
					if (nsubj.gov().nodeString().equals(cop.gov().nodeString())){
						Aspect aspect = new Aspect(nsubj.dep().nodeString(), cop.gov().nodeString());
						aspects.add(aspect);
					}
				}
				if(dobj != null){
					if (term.equals(dobj.gov().nodeString())){
						Aspect aspect = new Aspect(dobj.dep().nodeString(), term);
						aspects.add(aspect);
					}
				}
			}
			
			// Handle 5-9 patterns
			if (aspects.size() > 0) {
				for (Object obj : objs) {
					TypedDependency dependency = (TypedDependency) obj;
					String term = dependency.gov().nodeString();
					String modifier = dependency.dep().nodeString();
					if (dependency.reln().getShortName().startsWith("conj")) {
						for (Aspect asp : aspects) {
							if (term.equalsIgnoreCase(asp.getTerm())) {
								Aspect aspect = new Aspect(term, asp.getModifier());
								aspects.add(aspect);
								break;
							}
							if (term.equalsIgnoreCase(asp.getModifier())) {
								Aspect aspect = new Aspect(asp.getTerm(), modifier);
								aspects.add(aspect);
								break;
							}
						}
					} else if (dependency.reln().getShortName().equals("nn")) {
						for (Aspect asp : aspects) {
							if (term.equalsIgnoreCase(asp.getTerm())) {
								asp.setTerm(modifier + " " + asp.getTerm());
								break;
							}
							if (term.equalsIgnoreCase(asp.getModifier())) {
								asp.setTerm(term + " " + asp.getTerm());
								break;
							}
						}
					}
				}
			}
			
			// Handle negative word
			if (neg != null){
				for (Aspect asp : aspects) {
					if (asp.getModifier().equals(neg.gov().nodeString())){
						asp.setModifier("not_" + asp.getModifier());
					}
				}
			}
		}
		
		return aspects;
	}
	
	public static List<String> parseSentence(StanfordCoreNLP parser, String text) {
		Annotation annotation = parser.process(text);

		List<String> lines = new ArrayList<String>();
		for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			for (CoreLabel label : labels){
				String line = label.lemma() + "\t" + label.tag();
				lines.add(line);
			}
		}
		
		return lines;
	}
	
	public static String extractTokens(StanfordCoreNLP tagger, String text, String tag) {
        StringBuffer tokens = new StringBuffer();
        
        Annotation document = new Annotation(text);
        tagger.annotate(document);
        
        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        
        for (CoreMap sentence : sentences) {
            List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
            for (CoreLabel label : labels) {
                if (label.get(PartOfSpeechAnnotation.class).equals(tag)) {
                	tokens.append(label.lemma() + " ");
                }
            }
        }
        
        return tokens.toString();
    }
}
