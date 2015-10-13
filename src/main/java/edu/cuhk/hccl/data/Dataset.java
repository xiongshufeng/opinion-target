package edu.cuhk.hccl.data;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

public class Dataset {
	
	private List<Instance> instances;
	
	public Dataset(){
		instances = new ArrayList<Instance>();
	}
	
	public void loadData(String dataFile) throws IOException{
		String fileContent = FileUtils.readFileToString(new File(dataFile));
		String[] splits = fileContent.split("\\n\\n");
		for (String split : splits){
			Instance instance = new Instance();
			String[] lines = split.split("\n");
			for (String line : lines){
				Record record = new Record(line);
				instance.appendRecord(record);
			}
			instances.add(instance);
		}
	}
	
	public List<Instance> getInstances(){
		return instances;
	}
	
	public class Instance {
		
		List<Record> records;
		
		public Instance(){
			records = new ArrayList<Record>();
		}
		
		public void appendRecord(Record record){
			records.add(record);
		}
		
		public String getText(){
			StringBuffer buffer = new StringBuffer();
			
			for (Record record : records){
				buffer.append(record.text);
				buffer.append(" ");
			}
			
			return buffer.toString().trim();
		}
	}
}


class Record {
	
	String text;
	String pos;
	List<String> labels;
	
	public Record(){
		labels = new ArrayList<String>();
	}
	
	public Record(String line) {
		
		String[] splits = line.split(" ");
		text = splits[0];
		pos = splits[1];
		
		for (int i = 2; i < splits.length; i++)
			appendLabel(splits[i]);
	}

	public void appendLabel(String label){
		labels.add(label);
	}
}
