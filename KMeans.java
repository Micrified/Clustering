import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class KMeans extends ClusteringAlgorithm
{
	// Number of clusters
	private int k;

	// Dimensionality of the vectors
	private int dim;
	
	// Threshold above which the corresponding html is prefetched
	private double prefetchThreshold;
	
	// Array of k clusters, class cluster is used for easy bookkeeping
	private Cluster[] clusters;

	/// Random number generator.
	private Random random;
	
	// This class represents the clusters, it contains the prototype (the mean of all it's members)
	// and memberlists with the ID's (which are Integer objects) of the datapoints that are member of that cluster.
	// You also want to remember the previous members so you can check if the clusters are stable.
	static class Cluster
	{
		float[] prototype;

		Set<Integer> currentMembers;
		Set<Integer> previousMembers;
		  
		public Cluster()
		{
			currentMembers = new HashSet<Integer>();
			previousMembers = new HashSet<Integer>();
		}

		public void printDescription () {
			System.out.println("\n** Current Members **");
			printSet(currentMembers);
		}
	}
	// These vectors contains the feature vectors you need; the feature vectors are float arrays.
	// Remember that you have to cast them first, since vectors return objects.
	private Vector<float[]> trainData;
	private Vector<float[]> testData;

	// Results of test()
	private double hitrate;
	private double accuracy;


	
	public KMeans(int k, Vector<float[]> trainData, Vector<float[]> testData, int dim)
	{
		this.k = k;
		this.trainData = trainData;
		this.testData = testData; 
		this.dim = dim;
		this.random = new Random();
		prefetchThreshold = 0.5;
		
		// Here k new cluster are initialized
		clusters = new Cluster[k];
		for (int ic = 0; ic < k; ic++)
			clusters[ic] = new Cluster();
	}

	/******************************* PRINTING METHODS *****************************/

	/// Prints the state of all clusters.
	public void printClusters (Cluster[] clusters) {
		for (Cluster c : clusters) {
			c.printDescription();
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
		System.out.print("]");
	}

	/// Prints all information within a vector.
	public void printVector (Vector<float[]> vector) {
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
	public void printArray (float[] array) {
		System.out.print("[");
		for (int i = 0; i < array.length; i++) {
			System.out.format("%.0f", array[i]);
		}
		System.out.println("]");
	}

	/******************************* TRAINING METHODS *****************************/

	/// Returns a random permutation from zero to (n - 1)
	public int[] randomPermutation (int n) {
		
	}

	/// Partitions given data to random clusters.
	public void randomPartition (Vector<float[]> data, Cluster[] clusters) {
		for (int i = 0; i < data.size(); i++) {
			clusters[random.nextInt(clusters.length)].currentMembers.add(i);
		}
	}

	/// Stages all clusters for data assignment by moving current members to old.
	public void stageClustersForPointAssignment(Cluster [] clusters) {
		for (Cluster c : clusters) {
			c.stage();
		} 
	}

	/// Assigns datapoints to their nearest clusters using Euclidean distance.
	public void clusterDatapoints (int[] permutation, Vector<float[]> data, Cluster[] clusters) {
		int i, j, n = data.size();
		float v[];
		Cluster c;

		for (i = 0; i < n; i++) {
			v = data[permutation[i]];
			c = closestPrototype(v, clusters);

		}
	}


	/// Generates a list of numbers from 0 to x
	public int[] loop(int x) {
    	int[] a = new int[x];
    	for (int i = 0; i < x; ++i) {
       	  a[i] = i;
    	}
    	return a;
	}

	public boolean train()
	{

		/// Step 1: Partition training data to random clusters.
		randomPartition(this.trainData, this.clusters);
		printClusters(this.clusters);

		/// Step 2: Using a random permutation, iterate through data points and
		///			assign them to nearest prototype with Euclidean distance
		///			measure.
		//int[] permutation = randomPermutation(self.clusters.size());
		clusterDatapoints(permutation, self.trainData, self.clusters);

		/// Step 3: Recompute the mean position of the prototypes.

		/// Step 4: Check whether membership is stable (below threshold) to stop.


		// Step 1: Select an initial random partioning with k clusters
		//partitionClusters();


		//showMembers();


	 	//implement k-means algorithm here:
		// Step 1: Select an initial random partioning with k clusters
		// Step 2: Generate a new partition by assigning each datapoint to its closest cluster center
		// Step 3: recalculate cluster centers
		// Step 4: repeat until clustermembership stabilizes
		return false;
	}

	public boolean test()
	{
		// iterate along all clients. Assumption: the same clients are in the same order as in the testData
		// for each client find the cluster of which it is a member
		// get the actual testData (the vector) of this client
		// iterate along all dimensions
		// and count prefetched htmls
		// count number of hits
		// count number of requests
		// set the global variables hitrate and accuracy to their appropriate value
		return true;
	}


	// The following members are called by RunClustering, in order to present information to the user
	public void showTest()
	{
		System.out.println("Prefetch threshold=" + this.prefetchThreshold);
		System.out.println("Hitrate: " + this.hitrate);
		System.out.println("Accuracy: " + this.accuracy);
		System.out.println("Hitrate+Accuracy=" + (this.hitrate + this.accuracy));
	}
	
	public void showMembers()
	{
		for (int i = 0; i < k; i++)
			System.out.println("\nMembers cluster["+i+"] :" + clusters[i].currentMembers);
	}
	
	public void showPrototypes()
	{
		for (int ic = 0; ic < k; ic++) {
			System.out.print("\nPrototype cluster["+ic+"] :");
			
			for (int ip = 0; ip < dim; ip++)
				System.out.print(clusters[ic].prototype[ip] + " ");
			
			System.out.println();
		 }
	}

	// With this function you can set the prefetch threshold.
	public void setPrefetchThreshold(double prefetchThreshold)
	{
		this.prefetchThreshold = prefetchThreshold;
	}
}
