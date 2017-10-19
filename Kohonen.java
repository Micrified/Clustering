import java.util.*;
import java.awt.geom.Point2D.Float;

public class Kohonen extends ClusteringAlgorithm
{
	// Size of clustersmap
	private int n;

	// Number of epochs
	private int epochs;
	
	// Dimensionality of the vectors
	private int dim;
	
	// Threshold above which the corresponding html is prefetched
	private double prefetchThreshold;

	private double initialLearningRate; 
	
	// This class represents the clusters, it contains the prototype (the mean of all it's members)
	// and a memberlist with the ID's (Integer objects) of the datapoints that are member of that cluster.  
	private Cluster[][] clusters;

	// Vector which contains the train/test data
	private Vector<float[]> trainData;
	private Vector<float[]> testData;
	
	// Results of test()
	private double hitrate;
	private double accuracy;
	
	static class Cluster {

		/// Cluster position.
		Float coord;
		
		/// Prototype vector.
		float[] prototype;

		/// Current members of the cluster.
		Set<Integer> currentMembers;

		/// Cluster constructor assigns point, initializes random prototype.
		public Cluster (int x, int y, int dim) {
			coord = new Float(x, y);
			currentMembers = new HashSet<Integer>();
			prototype = randomVector(dim);
		}

		/// Updates prototype vector using learning rate and input vector.
		public void updatePrototype (double eta, float[] v) {
			int n = prototype.length;

			for (int i = 0; i < n; i++) {
				prototype[i] = (float)(prototype[i] * (1 - eta) + (eta * v[i]));
			}
		}

		/// Prints a description of the Cluster instance.
		public void printDescription () {
			int n = prototype.length;
			System.out.println("----------------------------------[ Cluster ]-----------------------------------");
			System.out.print("Coordinate:\t" + coord.toString() + "\n");
			System.out.format("Prototype:\t[%.5f, %.5f, %.5f, ... , %.5f, %.5f, %.5f]\n", prototype[0], prototype[1], prototype[2], prototype[n-3], prototype[n - 2], prototype[n - 1]);
			System.out.print("Curr Memb:\t"); printSet(currentMembers);
		}
	}
	
	public Kohonen(int n, int epochs, Vector<float[]> trainData, Vector<float[]> testData, int dim) {
		this.n = n;
		this.epochs = epochs;
		prefetchThreshold = 0.5;
		initialLearningRate = 0.8;
		this.trainData = trainData;
		this.testData = testData; 
		this.dim = dim;       
		
		Random rnd = new Random();

		// Here n*n new cluster are initialized
		clusters = new Cluster[n][n];
		for (int x = 0; x < n; x++)  {
			for (int y = 0; y < n; y++) {
				clusters[x][y] = new Cluster(x, y, dim);
			}
		}
	}

	/******************************* SUPPORT METHODS ******************************/

	/// Returns a binary float vector.
	static public float[] randomVector (int length) {
		int b, i = 0, j = 0, k = 0, size = (length / 8) + (length % 8 == 1 ? 1 : 0);
		byte[] bs = new byte[size]; new Random().nextBytes(bs);
		float[] vector = new float[length];

		/// Extract bits over byte sequence, assign as vector value.
		for (i = 0; i < size; i++) {
			b = bs[i];
			for (j = 0; j < 8; j++) {
				vector[k % length] = (float)(b & 1);
				b >>= 1;
				k++;
			}
		}
		return vector;
	}

	/// Euclidean distance between feature vectors 'a' and 'b'.
	public double euclideanDistance (float[] a, float[] b) {
		int n = (a.length < b.length ? a.length : b.length);
		double d = 0;

		for (int i = 0; i < n; i++) {
			d += (b[i] - a[i]) * (b[i] - a[i]);
		}

		return Math.sqrt(d);
	}

	/// Manhattan distance between two points.
	public double manhattanDistance (Float a, Float b) {
		return Math.abs(b.x - a.x) + Math.abs(b.y - a.y);
	}
	
	/// Finds all nodes (clusters) within range of the BMU (dumb search).
	Vector<Cluster> neighboursWithinDistance(double r, Cluster bmu, Cluster[][] clusters) {
		Vector<Cluster> neighbours = new Vector<Cluster>();

		/// Manhattan distance for diamond search area. 
		for (Cluster[] cs : clusters) {
			for (Cluster c : cs) {
				double d = manhattanDistance(bmu.coord, c.coord);
				if (d <= r) {
					neighbours.add(c);
				}
			}
		}

		return neighbours;
	}

	/******************************* TRAINING METHODS *****************************/

	/// Iterates through cluster grid, returns best matching unit.
	public Cluster findBMU (float[] v, Cluster[][] clusters) {
		Cluster bmu = null;
		double d, min = 10E10;

		for (Cluster[] cs : clusters) {
			for (Cluster c : cs) {
				if ((d = euclideanDistance(v, c.prototype)) < min) {
					bmu = c;
					min = d;
				}
			}
		}

		return bmu;
	}

	/// Updates all neighbouring nodes to the bmu with a new prototype value.
	public void updateNeighbours (float[] v, double r, double eta, Cluster bmu, Cluster[][] clusters) {
		Vector<Cluster> neighbours = neighboursWithinDistance(r, bmu, clusters);

		for (Iterator<Cluster> i = neighbours.iterator(); i.hasNext();) {
			i.next().updatePrototype(eta, v);
		}
	}

	/// Updates learning rate.
	public double updateLearningRate (double t, double epochs) {
		return 0.8 * (1 - t / epochs);
	}

	/// Update neighbourhood radius.
	public double updateSquareSize (double n, double t, double epochs) {
		return (n / 2) * (1 - t / epochs);
	}

	public boolean train()
	{
		Cluster bmu;
		double r = -1, eta = -1;
		float[] v;

		/* DEBUG */
		System.out.println("*************************** Step 1: Random Prototypes **************************");
		printClusters(this.clusters);
		System.out.println("***************************** Step (2,3): Training *****************************");

		/// Repeating 'epoch' times.
		for (int t = 0; t < epochs; t++) {

			/// Step 2: Compute squareSize, learningRate.
			r = updateSquareSize(n, t, epochs);
			eta = updateLearningRate(t, epochs);

			/* DEBUG */
			System.out.format("\n\nEpoch %d, r = %.5f, eta = %.5f\n\n", t, r, eta);

			/// Step 2: For all input vectors, find BMU.
			for (Iterator<float[]> i = trainData.iterator(); i.hasNext();) {
				v = i.next();
				bmu = findBMU(v, this.clusters);

				/// Step 3: Update all nodes within the neighbourhood of the BMU.
				updateNeighbours(v, r, eta, bmu, this.clusters);
			}

			/* DEBUG */
			printClusters(this.clusters);

			/// [Optional]: Print progress.
			/// System.out.format("Training at %.1f%%\n", ((float)t / (float) epochs) * 100.0);
		}

		/* DEBUG */
		System.out.format("\n\nCompleted %d epochs, r = %.5f, eta = %.5f\n\n", epochs, r, eta);
		
		return true;
	}
	
	public boolean test()
	{
		// iterate along all clients
		// for each client find the cluster of which it is a member
		// get the actual testData (the vector) of this client
		// iterate along all dimensions
		// and count prefetched htmls
		// count number of hits
		// count number of requests
		// set the global variables hitrate and accuracy to their appropriate value
		return true;
	}


	public void showTest()
	{
		System.out.println("Initial learning Rate=" + initialLearningRate);
		System.out.println("Prefetch threshold=" + prefetchThreshold);
		System.out.println("Hitrate: " + hitrate);
		System.out.println("Accuracy: " + accuracy);
		System.out.println("Hitrate+Accuracy=" + (hitrate + accuracy));
	}
 
 
	public void showMembers()
	{
		for (int i = 0; i < n; i++)
			for (int i2 = 0; i2 < n; i2++)
				System.out.println("\nMembers cluster["+i+"]["+i2+"] :" + clusters[i][i2].currentMembers);
	}

	public void showPrototypes()
	{
		for (int i = 0; i < n; i++) {
			for (int i2 = 0; i2 < n; i2++) {
				System.out.print("\nPrototype cluster["+i+"]["+i2+"] :");
				
				for (int i3 = 0; i3 < dim; i3++)
					System.out.print(" " + clusters[i][i2].prototype[i3]);
				
				System.out.println();
			}
		}
	}

	public void setPrefetchThreshold(double prefetchThreshold)
	{
		this.prefetchThreshold = prefetchThreshold;
	}

	/******************************* PRINTING METHODS *****************************/

	/// Prints the state of all clusters.
	public void printClusters (Cluster[][] clusters) {
		for (Cluster cs[] : clusters) {
			for (Cluster c : cs) {
				c.printDescription();
			}
		} 
	}

	/// Prints an integer set. 
	public static void printSet (Set<Integer> set) {
		System.out.print("[");
		Iterator<Integer> iterator = set.iterator();
		while (iterator.hasNext()) {
			Integer i = iterator.next();
			System.out.format("%d", i);
			if (iterator.hasNext() == true) {
				System.out.print(",");
			}
		}
		System.out.println("]");
	}

	/// Prints all information within a vector.
	public static void printVector (Vector<float[]> vector) {
		int m, n = vector.size();
		float[] v;

		for (int i = 0; i < n; i++) {
			v = vector.get(i);
			m = v.length;
			System.out.format("%d) [", i);
			for (int j = 0; j < m; j++) {
				System.out.format("%.0f", v[j]);
			}
			System.out.println("]");
		}
	}

	/// Prints a float array.
	public static void printArray (float[] array) {
		System.out.print("[");
		for (int i = 0; i < array.length; i++) {
			System.out.format("%.3f", array[i]);
			if (i < array.length - 1) {
				System.out.print(",");
			}
		}
		System.out.println("]");
	}
}

