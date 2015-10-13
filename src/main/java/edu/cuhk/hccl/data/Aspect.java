package edu.cuhk.hccl.data;

public class Aspect {
	
	protected String term;
	protected String modifier;

	public Aspect(String term, String modifier) {
		this.term = term;
		this.modifier = modifier;
	}
	
	public String getModifier() {
		return modifier;
	}

	public void setModifier(String modifier) {
		this.modifier = modifier;
	}

	public String getTerm() {
		return term;
	}

	public void setTerm(String term) {
		this.term = term;
	}

}
