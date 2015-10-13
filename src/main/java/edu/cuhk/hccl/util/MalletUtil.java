package edu.cuhk.hccl.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cc.mallet.classify.Classifier;

public class MalletUtil {
	
	public static Classifier loadClassifer (File modelFile) 
			throws FileNotFoundException, IOException, ClassNotFoundException{
		
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFile));
		Classifier classifier = (Classifier)ois.readObject();
		ois.close();
		
		return classifier;	
	}
	
	public static void saveClassifer (Classifier classifier, File modelFile) 
			throws IOException{
		
		if (modelFile.exists()){
			modelFile.delete();
		}
		
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFile));
		oos.writeObject(classifier);
		oos.close();
	}
}
