package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.xml.bind.JAXBException;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.FileUtils;

public class AppSplitDataset {

	public static void main(String[] args) throws JAXBException, IOException, ParseException {

		System.out.println("[INFO] Processing is started...");

		CommandLineParser parser = new BasicParser();
		Options options = new Options();

		options.addOption("t", "train", true, "Train file");
		options.addOption("r", "ratio", true, "Split ratio of development and validation");
		options.addOption("d", "development", true, "Development set");
		options.addOption("v", "validation", true, "Validation set");
		
		CommandLine line = parser.parse(options, args);

		// Get parameters
		String trainFile = line.getOptionValue('t');
		float devRatio = Float.parseFloat(line.getOptionValue('r'));
		String devFile = line.getOptionValue('d');
		String valFile = line.getOptionValue('v');
		
		// Read the whole trainSet
		List<String> trainSet = FileUtils.readLines(new File(trainFile));
		
		// Split trainSet into devSet and valSet
		List<String> devSet = new ArrayList<String>();
		List<String> valSet = new ArrayList<String>();
		
		int devSize = (int) (trainSet.size() * devRatio);
		
		List<String> record = new ArrayList<String>();
		for (String strLine : trainSet){
			record.add(strLine);
			if (strLine.isEmpty()) {
				double random = Math.random();
				if (random <= devRatio && devSet.size() < devSize) {
					devSet.addAll(record);
				} else {
					valSet.addAll(record);
				}
				
				record.clear();
			}
		}
		
		FileUtils.writeLines(new File(devFile), devSet, false);
		FileUtils.writeLines(new File(valFile), valSet, false);
		
		System.out.println("[INFO] Development set is written to: " + devFile);
		System.out.println("[INFO] Validation set is written to: " + valFile);
		System.out.println("[INFO] Processing is finished!");
	}
	
}
