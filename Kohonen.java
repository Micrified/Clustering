import java.util.*;

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
	
	static class Cluster
	{
			float[] prototype;

			int x, y;

			Set<Integer> currentMembers;

			public Cluster()
			{
				currentMembers = new HashSet<Integer>();
			}

			/// initialize randomly
			public void initialize(){
				
			}

			
	}
	
	public Kohonen(int n, int epochs, Vector<float[]> trainData, Vector<float[]> testData, int dim)
	{
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
		for (int i = 0; i < n; i++)  {
			for (int i2 = 0; i2 < n; i2++) {
				clusters[i][i2] = new Cluster();
				clusters[i][i2].prototype = new float[dim];
				/// initialize the prototype randomly
				float[] currentPrototypes = clusters[i][i2].getFloat();
				for ( int idx = 0; idx < dim; idx++){
					clusters[i][i2].setPrototype(idx) = Math.round(Math.random()) (float);
				}
				
			}
		}
	}

	/// Method that calculates the euclidean distance between two vectors
	public float euclideanDistance(float[] vec1, float[] vec2)
	{
		int sum = 0;
		for( int i = 0; i < vec1.size(); i++)
		{
			sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
		}
		return Math.sqrt(sum);
	}

	/// Find the best matching unit (BMU)
	public Cluster findBMU()
	{	
		/// Find BMU by iterating through all prototypes
		float minDist = Float.MAX_VALUE;
		Cluster BMU;
		for( int protIdx1 = 0; protIdx1 < this.n; protIdx1++ )
		{
			for( int protIdx2 = 0; protIdx2 < this.n; protIdx2++ )
			{
				float distToProt = euclideanDistance(trainVec,clusters[protIdx1][protIdx2])
				if( dist < minDist)
						{
					minDist = dist;	
					BMU = clusters[protIdx1][protIdx2];
					BMU.x  = protIdx1;
					BMU.y = protIdx2;
				}
			{
		}
		return BMU;
	}

	/// Adjust all prototypes in the neighbourhood of the BMU
	public void adjustNeighbourhood(Cluster BMU, double radius, double learnRate)
	{
		for( int protIdx1 = 0; protIdx1 < this.n; protIdx1++ )
		{
			for( int protIdx2 = 0; protIdx2 < this.n; protIdx2++ )
			{
				/// If a prototype is within the radius, adjust it
				double dist = (protIdx1-BMU.x)*(protIdx1-BMU.x) + (protIdx2-BMU.y)*(protIdx2-BMU.y);
				if ( dist <= radius * radius) 
				{
					clusters[protIdx1][protIdx2] = (1-learnRate) * clusters[protIdx1][protIdx2] + learnRate * trainVec;
				}
	}

	public boolean train()
	{

		/// Repeat 'epochs' times:
		for( int t = 0; t < this.epochs; t++)
		{	
			if( this.epochs % t
			/// Calculate current learning rate and radius
			double learnRate = 0.8 * (1 - (t / this.epochs));
			double radius = this.n / 2 * (1 - (t / this.epochs));
			/// Iterate through all training points
			for( int trainIdx = 0; trainIdx < trainData.size(); trainIdx++ )
			{
				float[] trainVec = trainData.get(trainIdx);
				Cluster BMU = findBMU();
				adjustNeighbourhood(BMU, radius, learnRate);
				
			}
			
	
		
			// For each vector its Best Matching Unit is found, and :
				// Step 4: All nodes within the neighbourhood of the BMU are changed, you don't have to use distance relative learning.
		}
			
		// Since training kohonen maps can take quite a while, presenting the user with a progress bar would be nice
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
}

