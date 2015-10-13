package edu.cuhk.hccl.util;

import java.io.File;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;

import edu.cuhk.hccl.data.Sentences;

public class XmlUtil {

	public static Sentences parseXML(String xmlFile) throws JAXBException {
		JAXBContext context = JAXBContext.newInstance(Sentences.class);
		Unmarshaller unmarshaller = context.createUnmarshaller();
		Sentences sents = (Sentences) unmarshaller.unmarshal(new File(xmlFile));
		return sents;
	}

}
