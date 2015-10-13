package edu.cuhk.hccl;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.GZIPInputStream;

public class GoogleWordEmbedding extends WordEmbedding {

	private static final int MAX_SIZE = 50;

	public GoogleWordEmbedding(String modelPath) {
		super(modelPath);
	}

	/***
	 * The following methods are based on the code gist from
	 * https://gist.github.com/ansjsun/6304960
	 * 
	 * @param path
	 * @throws IOException
	 */
	@Override
	public void loadWordVectors() throws IOException {

		DataInputStream dis = null;
		BufferedInputStream bis = null;

		try {
			bis = new BufferedInputStream(new GZIPInputStream(new FileInputStream(modelPath)));
			dis = new DataInputStream(bis);
			int words = Integer.parseInt(readString(dis));
			int size = Integer.parseInt(readString(dis));

			float[] vectors = null;
			for (int i = 0; i < words; i++) {
				String word = readString(dis);
				vectors = new float[size];
				double len = 0;
				for (int j = 0; j < size; j++) {
					float vector = readFloat(dis);
					len += vector * vector;
					vectors[j] = (float) vector;
				}
				len = Math.sqrt(len);

				for (int j = 0; j < vectors.length; j++) {
					vectors[j] = (float) (vectors[j] / len);
				}
				wordVecMap.put(word, vectors);
				System.out.println(String.format("Loading vector for word: %s", word));
			}

		} finally {
			bis.close();
			dis.close();
		}
	}

	private static float bytesToFloat(byte[] b) {
		int accum = 0;
		accum = accum | (b[0] & 0xff) << 0;
		accum = accum | (b[1] & 0xff) << 8;
		accum = accum | (b[2] & 0xff) << 16;
		accum = accum | (b[3] & 0xff) << 24;
		return Float.intBitsToFloat(accum);
	}

	private static String readString(DataInputStream dis) throws IOException {

		byte[] bytes = new byte[MAX_SIZE];
		byte b = dis.readByte();
		int i = -1;
		StringBuilder sb = new StringBuilder();
		while (b != 32 && b != 10) {
			i++;
			bytes[i] = b;
			b = dis.readByte();
			if (i == 49) {
				sb.append(new String(bytes));
				i = -1;
				bytes = new byte[MAX_SIZE];
			}
		}
		sb.append(new String(bytes, 0, i + 1));
		return sb.toString();
	}

	private static float readFloat(InputStream is) throws IOException {

		byte[] bytes = new byte[4];
		is.read(bytes);
		return bytesToFloat(bytes);
	}

}
