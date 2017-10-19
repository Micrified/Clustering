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

			/// variables to store the position of the BMU in the map
			int x, y;

			Set<Integer> currentMembers;

			public Cluster()
			{
				currentMembers = new HashSet<Integer>();
			}

			public float[] getPrototype()
			{
				return prototype;
			}

			public void setPrototype(float[] newPrototype)
			{
				prototype = newPrototype;
			}

			public void setPrototype(int idx, float value)
			{
				prototype[idx] = value;
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
		for (int i = 0; i < n; i++)  
		{
			for (int i2 = 0; i2 < n; i2++) 
			{
				clusters[i][i2] = new Cluster();
				clusters[i][i2].prototype = new float[dim];
				/// initialize the prototypes by assigning a random number between 0 to 1 to each feature
				float[] currentPrototypes = clusters[i][i2].getPrototype();
				for ( int idx = 0; idx < dim; idx++)
				{
                   			 clusters[i][i2].setPrototype(idx, rnd.nextFloat() );
				}
				
			}
		}
	}

	/************************* TRAINING METHODS ****************************/

	/// Method that calculates the euclidean distance between two float arrays
	public double euclideanDistance(float[] vec1, float[] vec2)
	{
		double sum = 0;
		for( int i = 0; i < vec1.length; i++)
		{
			sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
		}
		return Math.sqrt(sum);
	}

	/// Method that finds the best matching unit (BMU)
	public Cluster findBMU(float[] client)
	{	
		/// Find BMU by iterating through all prototypes
		/// BMU is the prototype that is closest to the client array vector
		double minDist = Double.MAX_VALUE;
		Cluster BMU = clusters[0][0];
		for( int protIdx1 = 0; protIdx1 < this.n; protIdx1++ )
		{
			for( int protIdx2 = 0; protIdx2 < this.n; protIdx2++ )
			{
				double distToProt = euclideanDistance(client, clusters[protIdx1][protIdx2].getPrototype());
				if (distToProt < minDist)
				{
					minDist = distToProt;
					BMU = clusters[protIdx1][protIdx2];
					BMU.x = protIdx1;
					BMU.y = protIdx2;
				}
			}
		}
		return BMU;
	}

	/// Adjust single cluster to make it more similar to the training vector
	public void adjustCluster(double learnRate, Cluster cluster, float[] trainVec)
	{
	    float[] prototypeOfCluster = cluster.getPrototype();
        for(int i = 0; i < trainVec.length; i++)
        {
            prototypeOfCluster[i] = (float) ((1-learnRate) * prototypeOfCluster[i] + learnRate * trainVec[i]);
        }
        cluster.setPrototype(prototypeOfCluster);
	}


	/// Adjust all prototypes in the neighbourhood of the BMU
	public void adjustNeighbourhood(Cluster BMU, double radius, double learnRate, float[] trainVec)
	{
		for( int protIdx1 = 0; protIdx1 < this.n; protIdx1++ )
		{
			for( int protIdx2 = 0; protIdx2 < this.n; protIdx2++ )
			{
				/// If a prototype is within the radius and therefore in the neighbourhood, adjust it
				int manHatDist = Math.abs(protIdx1-BMU.x) + Math.abs(protIdx2-BMU.y);
				if ( manHatDist <=  radius)
				{
					Cluster currentCluster = clusters[protIdx1][protIdx2];
					adjustCluster(learnRate,currentCluster, trainVec);
				}
			}
		}
	}

	public boolean train()
	{
		/// Repeat 'epochs' times:
		for( int t = 0; t < this.epochs; t++)
		{
            		/// Print progress in percentages
			System.out.print("\r[");
			System.out.print(Math.round((1000.0*t)/this.epochs)/10.0+"%]");

			/// Calculate current learning rate and radius
			float learnRate = (float) 0.8 * (1 - (t / this.epochs));
			double radius = this.n / 2 * (1 - (t / this.epochs));

			/// Iterate through all training points. Find BMU for each training point and adjust BMU's neighbourhood
			for( int trainIdx = 0; trainIdx < trainData.size(); trainIdx++ )
			{
				float[] trainVec = trainData.get(trainIdx);
				Cluster BMU = findBMU(trainVec);
				adjustNeighbourhood(BMU, radius, learnRate, trainVec);
			}
		}
		System.out.println();

		/// Add train data to membership sets of clusters, mostly for visualization purposes
		/// This membership assignment is not necessary, as it can be also calculated on the go in the test phase
		for(int i = 0; i < trainData.size(); i++)
		{
		    Cluster closestPrototype = findBMU(trainData.get(i));
		    closestPrototype.currentMembers.add(i);
		}

		return true;
	}

	/********************** TEST METHODS ****************/

	/// Find the prototype that contains the member of which the idx is given
	public float[] findAssignedPrototype(int idxOfInput)
        {
	    for( int i = 0; i < this.n; i++ )
	    {
	        for( int j = 0; j < this.n; j++ )
	        {
	            if(clusters[i][j].currentMembers.contains(idxOfInput))
	                return clusters[i][j].getPrototype();
	        }
	    }
	    System.out.println("ERROR: training vector "+ idxOfInput + " was not assigned to a prototype");
	    System.exit(0);
	    return null;
        }

	public boolean test()
	{
	    double requests = 0, prefetched = 0, hits = 0;
            // iterate along all clients
            for( int i = 0; i < testData.size(); i++)
            {
                /// Find the closest cluster
                float[] assignedPrototype = findAssignedPrototype(i);
                float[] testDataClient = testData.get(i);

                for (int j = 0; j < dim; j++)
                {
                    Boolean wasPrefetched = (assignedPrototype[i] > prefetchThreshold);
                    Boolean requested = (testDataClient[i] != 0);

                    /// Add to requests.
                    requests += (requested ? 1 : 0);

                    /// Add to prefetch
                    prefetched += (wasPrefetched ? 1 : 0);

		    /// If something was requested and fetched, then we have a true positive, a hit
                    if (requested && wasPrefetched)
                        hits++;
                }
            }
            hitrate = hits/requests;
            accuracy = hits/(prefetched);
            showTest();
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

