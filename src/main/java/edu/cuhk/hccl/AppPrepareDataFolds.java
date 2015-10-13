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

public class AppPrepareDataFolds {

	public static void main(String[] args) throws JAXBException, IOException, ParseException {

		System.out.println("[INFO] Processing is started...");

		CommandLineParser parser = new BasicParser();
		Options options = new Options();

		options.addOption("f", "file", true, "data file");
		options.addOption("n", "folds", true, "number of folds");
		options.addOption("d", "directory", true, "target direcory");
		
		CommandLine line = parser.parse(options, args);

		// Get parameters
		String dataFile = line.getOptionValue('f');
		int folds = Integer.parseInt(line.getOptionValue('n'));
		String targetDir = line.getOptionValue('d');
		
		// Read the whole trainSet
		List<String> dataLines = FileUtils.readLines(new File(dataFile));
		
		// Count the total number of records
		int counter = 0;
		for (String strLine : dataLines){
			if (strLine.isEmpty())
				counter++;
		}
		
		final int foldSize = counter / folds;
		final int numTests = foldSize;
		final int numTrains = counter - numTests ;
		System.out.println(String.format("Total size: %d; Train size: %d; Test size: %d", counter, numTrains, numTests));
		
		// Create trainSet and testSet
		for (int i = 0; i < folds; i++){
			
			int testStart = foldSize * i;
			int testEnd = testStart + numTests;
			System.out.println(String.format("Fold: %d; Test-Start: %d; Test-End: %d", i, testStart, testEnd - 1));
			System.out.println("==========");
			List<String> trainSet = new ArrayList<String>();
			List<String> testSet = new ArrayList<String>();
			
			counter = 0;
			List<String> record = new ArrayList<String>();
			for (String strLine : dataLines) {
				record.add(strLine);
				if (strLine.isEmpty()) {
					counter++;

					if (testStart <= counter && counter <= testEnd)
						testSet.addAll(record);
					else
						trainSet.addAll(record);
					
					record.clear();
				}
			}
			
			File trainFile = new File(targetDir + "/train" + i + ".tsv");
			File testFile = new File(targetDir + "/test" + i + ".tsv");
			
			FileUtils.writeLines(trainFile, trainSet, false);
			FileUtils.writeLines(testFile, testSet, false);
			
			System.out.println("[INFO] Train set is written to: " + trainFile.getPath());
			System.out.println("[INFO] Test set is written to: " + testFile.getPath());
		}
		
		System.out.println("[INFO] Processing is finished!");
	}
	
}
