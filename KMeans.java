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
		  
		public Cluster(int dim)
		{
			currentMembers = new HashSet<Integer>();
			previousMembers = new HashSet<Integer>();
			prototype = new float[dim];
		}

		/// Returns true if the cluster contains given index.
		public Boolean containsIndex (int n) {
			for (Integer i : currentMembers) {
				if (i.intValue() == n) {
					return true;
				}
			}
			return false;
		}

		/// Computes the mean (prototype) feature vector.
		public void updatePrototype (Vector<float[]> data) {
			int n = prototype.length, d = currentMembers.size();
			prototype = new float[n];

			/// Iterate across all member vectors, add up values.
			for (Integer i : currentMembers) {
				float[] v = data.elementAt(i.intValue());
				for (int j = 0; j < n; j++) {
					prototype[j] += v[j];
				}
			}

			/// Divide out by total members size to get mean.
			for (int i = 0; i < n; i++) {
				prototype[i] /= d;
			}
		}

		/// Computes the number of differences betweeen the previous and current members.
		public int membershipChanges() {
			int changed = 0;

			for (Integer i : currentMembers) {
				changed += (previousMembers.contains(i) == true) ? 0 : 1;
			}

			return changed;
		}

		/// Prints a description of the Cluster instance.
		public void printDescription () {
			System.out.println("----------------------------------[ Cluster ]-----------------------------------");
			System.out.print("Prototype:\t"); printArray(prototype);
			System.out.print("Curr Memb:\t"); printSet(currentMembers);
			System.out.print("Prev Memb:\t"); printSet(previousMembers);
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
			clusters[ic] = new Cluster(dim);
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

	/******************************* TRAINING METHODS *****************************/

	/// Partitions given data to random clusters.
	public void randomPartition (Vector<float[]> data, Cluster[] clusters) {
		for (int i = 0; i < data.size(); i++) {
			clusters[random.nextInt(clusters.length)].currentMembers.add(i);
		}
	}

	/// Returns a random permutation from zero to (n - 1)
	public int[] randomPermutation (int n) {
		int a, b, t, i;
		int[] permutation = new int[n];
		for (i = 0; i < n; permutation[i] = i, i++);
		for (i = 0; i < n; i++) {
			a = random.nextInt(n);
			b = random.nextInt(n);
			t = permutation[a];
			permutation[a] = permutation[b];
			permutation[b] = t;
		}
        return permutation;
	}

	/// Computes the total membership changes across clusters.
	public int totalMembershipChanges (Cluster [] clusters) {
		int changes = 0;

		for (Cluster c : clusters) {
			changes += c.membershipChanges();
		}

		return changes;
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

	/// Returns the closest prototype to the given feature vector.
	public Cluster closestPrototype(float[] v, Cluster[] clusters) {
		double d, min = 10E10;
		Cluster p = null;

		for (Cluster c : clusters) {
			if ((d = euclideanDistance(v, c.prototype)) < min) {
				min = d;
				p = c;
			}
		}

		return p;
	}

	/// Assigns datapoints to their nearest clusters using Euclidean distance.
	public void performClustering (int[] permutation, Vector<float[]> data, Cluster[] clusters) {
		int i, j, n = data.size();
		float v[];

		/// Move current members to previous, reset current members.
		for (Cluster c : clusters) {
			c.previousMembers = c.currentMembers;
			c.currentMembers = new HashSet<Integer>();
		}

		/// Assign datapoints to clusters.
		for (i = 0; i < n; i++) {
			v = data.elementAt(permutation[i]);
			Cluster c = closestPrototype(v, clusters);
			c.currentMembers.add(permutation[i]);
		}
	}

	/// Recomputes the prototypes for all clusters.
	public void recomputeMeanPositions(Vector<float[]> data, Cluster[] clusters) {
		for (Cluster c : clusters) {
			c.updatePrototype(data);
		}
	}

	public boolean train()
	{
		//implement k-means algorithm here:
		// Step 1: Select an initial random partioning with k clusters
		// Step 2: Generate a new partition by assigning each datapoint to its closest cluster center
		// Step 3: recalculate cluster centers
		// Step 4: repeat until clustermembership stabilizes

		/// Step 1: Partition training data to random clusters.
		randomPartition(this.trainData, this.clusters);
		recomputeMeanPositions(this.trainData, this.clusters);

		/* DEBUG */
		System.out.println("****************************** Step 1: Partitions ******************************");
		printClusters(this.clusters);
		System.out.println("***************************** Step (2,3): Training *****************************");
		int delta = 0;
		int round = 0;

		do {
			/* DEBUG */
			System.out.format("\n\nRound %d, Changed (previous cycle) = %d\n\n", round, delta);
			round++;

			/// Step 2: Obtain random permutation, reassign datapoints to clusters.
			int[] indexPermutation = randomPermutation(this.trainData.size());
			performClustering(indexPermutation, this.trainData, this.clusters);

			/// Step 3: Recompute mean positions of prototypes.
			recomputeMeanPositions(this.trainData, this.clusters);

			/* DEBUG */
			printClusters(this.clusters);

		} while ((delta = totalMembershipChanges(this.clusters)) > 0);

		/* DEBUG */
		System.out.format("\n\nStopped on round %d, Changed (previous cycle) = %d\n\n", round, delta);

		return false;
	}


	/****************************** TESTING METHODS *******************************/

	/// Returns a list of all Clusters containing indices 0 -> n.
	Cluster[] getAssignedClusters (int n , Cluster[] clusters) {
		Cluster[] assignedClusters = new Cluster[n];

		for (int i = 0; i < n; i++) {
			for (Cluster c : clusters) {
				if (c.containsIndex(i)) {
					assignedClusters[i] = c;
					break;
				}
			}
		}
		return assignedClusters;
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

		/// Step 1: Iterate through all clients, and find their corresponding clusters.
		int n = this.testData.size();
		Cluster[] assignedClusters = getAssignedClusters(n, this.clusters);

		/// Step 2: Iterate through all clients, along with their corresponding clusters.
		///			Count prefetched, hits, requests.
		int hits = 0, requests = 0;
		for (int i = 0; i < n; i++) {
			float[] v = this.testData.elementAt(i);
			float[] p = assignedClusters[i].prototype;

			for (int j = 0; j < dim; j++) {
				Boolean wasPrefetched = (p[i] > prefetchThreshold);
				prefetched += (wasPrefetched == true ? 1 : 0);
				
				if ((v[i] == 0) ^ wasPrefetched) {
					hits++;
				}
			}

		}

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
