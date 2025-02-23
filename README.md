# Min-Max Ant System applied to the Police Patrol Routing Problem

A final year project for the University of Exeter, using the Min-Max Ant System to solve the Police Patrol Routing Problem.

Scored 80/100 for this paper.

Supervisor: Dr Ayah Helal

## Project Description

[Demonstration Video](https://youtu.be/AYY5L_opb18?si=0pshWZoEI2HxabKp)

The enduring worry over crime remains a top priority for the public. It is unsurprising, given the well-documented links between crime rates, how people perceive crime, and their resulting influence on overall life satisfaction. As indicated by the Crime Survey for England and Wales (CSEW) report released by the Office for National Statistics (ONS), the United Kingdom encountered around 8.4 million offences up to June 2023. Police agencies have recently been given additional funding of £843 million with the task of improving productivity. While governments devote substantial resources to tackle crime, it becomes clear that the remedy does not solely lie in bolstering police numbers but rather in improving police effectiveness. This report explores the use of MMAS to address the limitations of police patrol routing, a problem commonly known as the Police Patrol Routing Problem (PPRP). The study focuses on examining the quality of solutions from applying an implementation of MMAS integrated with ABM applied to the PPRP. In this approach, a fleet of police patrol cars operates autonomously, coordinating within a shared environment to enhance resource allocation efficiency and boost the perceived police presence.

## Installation

To run the project, you will need the following dependencies:

- **Python 3.11**: The programming language used for development.
- **Jupyter Notebook**: Interactive computing environment.
- **NumPy**: Numerical computing library for handling arrays and matrices.
- **Matplotlib**: Plotting library for creating visualizations.
- **NetworkX**: Library for creating, analyzing, and visualizing complex networks.
- **Pandas**: Data manipulation and analysis library.
- **SciPy**: Library for scientific computing and mathematical functions.
- **Scikit-learn (Sklearn)**: Machine learning library for data analysis and modeling.
- **Geopandas**: Geospatial data manipulation library built on top of Pandas.
- **Shapely**: Library for geometric operations on objects in Python.
- **Folium**: Library for creating interactive maps.
- **OS**: Operating system interface.
- **CSV**: Library for reading and writing CSV files.
- **Re (Regular Expressions)**: Library for pattern matching and string manipulation.
- **Ast**: Library for Abstract Syntax Trees (AST) manipulation.
- **Random**: Library for generating random numbers.
- **String**: Library for string operations.
- **Time**: Library for time-related functions.
- **Requests**: HTTP library for making requests.
- **JSON**: Library for working with JSON data.
- **Statistics**: Library for statistical functions and calculations.

To install the dependencies, start by creating a virtual environment:

### Create a virtual environment named 'env'

python -m venv env

### Activate the virtual environment

### For Windows

.\env\Scripts\activate

### For macOS/Linux

source env/bin/activate

### Install the dependencies using pip

pip install "name of the dependency"

### Run the Jupyter Notebook

jupyter notebook

## Usage

### Using the Preprocessed Data

For convenience, the data has been preprocessed and stored in the folder `Cluster_edge_data_200` as CSV files. This data can be used to create the environment for the simulation. The data includes the origin and destination coordinates and the distance between the two points.

To create the environment, run the following code in the Jupyter Notebook labelled as Section 2: Environment Creation.

There are two options for creating the environment, the first option is to create the 200 cluster environment and the second option is to create the 20 cluster environment. These are clearly labelled and can be run by selecting the desired option.

Once the environment has been created, the simulation can be run using the MMAS algorithm. The MMAS algorithm is implemented in the Jupyter Notebook labelled as Section 3: MMAS Algorithm. There are two options for running the MMAS algorithm, the first option is to run the MMAS algorithm on the 200 cluster environment and the second option is to run the MMAS algorithm on the 20 cluster environment. These are clearly labelled and can be run by selecting the desired option.

In Section 3.1: Parameter tuning using Grid Search, the grid search algorithm is implemented to find the optimal parameters for the MMAS algorithm by tuning it on the 20 cluster environment. The results of the grid search algorithm are displayed below in Section 3.2: optimal parameters. The optimal parameters found by the grid search algorithm are then used to run the MMAS algorithm on the 200 cluster environment found in Section 3.3: MMAS Algorithm with optimal parameters. Each optimal parameter combination for the different number of agents is run 5 times and the results are averaged to find the average "global idle time" to ensure the results are consistent.

In Section 4: Random Walk Algorithm the random walk algorithm is implemented to compare the results of the MMAS algorithm. The random walk algorithm is run on the 200 cluster environment 5 times for each different number of agents and the results are averaged to find the average "global idle time" to ensure the results are consistent.

In Section 5: ACO Algorithm the ACO algorithm is implemented to compare the results of the MMAS algorithm. There are two options for running the ACO algorithm, the first option is to run the ACO algorithm on the 200 cluster environment and the second option is to run the ACO algorithm on the 20 cluster environment. These are clearly labelled and can be run by selecting the desired option. In Section 5.1: Parameter tuning using Grid Search, the grid search algorithm is implemented to find the optimal parameters for the ACO algorithm by tuning it on the 20 cluster environment. The results of the grid search algorithm are displayed below in Section 5.2: optimal parameters. The optimal parameters found by the grid search algorithm are then used to run the ACO algorithm on the 200 cluster environment found in Section 5.3: ACO Algorithm with optimal parameters. Each optimal parameter combination for the different number of agents is run 5 times and the results are averaged to find the average "global idle time" to ensure the results are consistent.

In Section 6: Results the results of the MMAS algorithm, Random Walk algorithm and ACO algorithm are displayed in a table and graph format. The results are compared to find the best algorithm for the Police Patrol Routing Problem.

To summarise, to use the preprocessed data, follow these steps:

1. Create the environment using the preprocessed data by running the code in Section 2: Environment Creation.
2. Tune the MMAS algorithm parameters on the 20 cluster environment by running the code in Section 3.1: Parameter tuning using Grid Search.
3. View the optimal parameters found by the grid search algorithm in Section 3.2: optimal parameters.
4. Run the MMAS algorithm on the 200 cluster environment with the optimal parameters found in Section 3.3: MMAS Algorithm with optimal parameters.
5. Get the results of the Random Walk algorithm by running the code in Section 4: Random Walk Algorithm for comparison with the MMAS algorithm.
6. Tune the ACO algorithm parameters on the 20 cluster environment by running the code in Section 5.1: Parameter tuning using Grid Search.
7. View the optimal parameters found by the grid search algorithm in Section 5.2: optimal parameters.
8. Run the ACO algorithm on the 200 cluster environment with the optimal parameters found in Section 5.3: ACO Algorithm with optimal parameters.
9. View the results of the MMAS algorithm, Random Walk algorithm and ACO algorithm in Section 6: Results.

### Using the Raw Data

If you would like to use the raw data, you can download the data from the following sources:

Crime Data: [UK Police Data](https://data.police.uk/data/)
Google Distance Matrix API: [Google Distance Matrix API](https://developers.google.com/maps/documentation/distance-matrix/overview)
Google GeoCoding API: [Google GeoCoding API](https://developers.google.com/maps/documentation/geocoding/overview)
Police Station Location Data: [Police Station Locations](https://www.cityoflondon.police.uk/a/your-area/city-of-london/city-of-london/community-policing/)
City of London Borough Boundaries: [City of London Borough Boundaries](https://skgrange.github.io/data.html)
You will also require a Google API key to access the Google Distance Matrix API and Google GeoCoding API. You can obtain an API key by following the instructions on the Google Cloud Platform website.

Once the data has been downloaded, you can preprocess the Crime Data by running the code in Section 1.1: Extract Crime Records from Dataset. This code will extract the crime records from the dataset and return them as a list of tuples containing the latitude and longitude of the crime location and the crime type.

In Section 1.2: Get the Different Crime Classifications used by the police the different crime classifications used by the police are extracted from the dataset and displayed.

In Section 1.3: Assign different weightings to each type of crime classification according to it's severity the different crime classifications are assigned different weightings according to their severity.

In Section 1.4: Plot the Coordinates on a Scatter Plot the coordinates of the crime locations are plotted on a scatter plot to visualise the data. After visualising the data, it can be seen that there are quite a few outliers.

In Section 1.5: Visualise the coordinates on the map the coordinates of the crime locations are plotted on a map to visualise the data. The outliers can be seen more clearly on the map.

In 1.6: Remove the outliers using Z-score the crime locations are clustered using the K means algorithm to group the crime locations into clusters. Each cluster is then assigned a centroid which will be used as the origin and destination for the police patrol cars. Each centroid is calculated as the mean of the coordinates of the crime locations in the cluster. Following that, the outliers are removed from the data using the Z-score method. However, this method isn't the best method to remove spatial outliers as z-score measures the distance of a point from the mean in terms of standard deviations. This doesn't hold true for spatial data as the distance between two points is not measured in standard deviations. Because the outliers are outliers based on spatial context, it would be better to remove them using geographical boundaries instead.

In Section 1.7: Add London Ward Borders to map the London ward borders are added to the map to visualise the geographical boundaries of London.

In Section 1.8: Remove all the borders except the city of london border the borders of London are removed from the map except for the City of London border. This is done to remove the outliers from the data using geographical boundaries found in Section 1.81. In Section 1.82: Create new map & display filtered points the outliers are removed from the data using the geographical boundaries of the City of London. The data is then displayed on a map to visualise the filtered data. In Section 1.83: Plot the clusters on a scatter plot the crime data points are clustered into 200 or 20 clusters using the K means algorithm. The clusters are then plotted on a scatter plot to visualise the data. In Section 1.84: Calculate the total severity scores for each cluster the total severity scores for each cluster are calculated by summing the severity scores of the crime classifications in each cluster. Each cluster is also given a cluster_no value as an index, the data is then saved into csv files to enable reconstruction of the same clusters. This is done for both 200 and 20 cluster environments. Section 1.85 Reconstruct Clustered data from files & remove clusters whose total_severity < 100 the clustered data is reconstructed from the csv files and clusters whose total severity is less than 100 are removed. This is done to ensure that only clusters of a certain severity level is used in the simulation.

In Section 1.9: Batching the data for the Google Distance Matrix API the data is batched into groups of 10 to be used in the Google Distance Matrix API. This is done for both the 200 and 20 cluster environments. Section 1.91: Get distances between Clusters using Google API the distances between the clusters are calculated using the Google Distance Matrix API. The geographical coordinates of the police stations within the City of London are also obtained using the Google GeoCoding API and the distances between the clusters and the police stations are calculated. The data is then saved into csv files to be used in the simulation. This is done for both the 200 and 20 cluster environments.

Following this the steps are the same as using the preprocessed data.

To summarise, to use the raw data, follow these steps:

1. Preprocess the Crime Data by running the code in Section 1.1: Extract Crime Records from Dataset.
2. Get the different crime classifications used by the police by running the code in Section 1.2.
3. Assign different weightings to each type of crime classification according to it's severity by running the code in Section 1.3.
4. Plot the coordinates on a scatter plot by running the code in Section 1.4.
5. Visualise the coordinates on the map by running the code in Section 1.5.
6. Add London Ward Borders to map by running the code in Section 1.7.
7. Remove all the borders except the city of london border by running the code in Section 1.8.
8. Remove the outliers using geographical boundaries by running the code in Section 1.81.
9. Create new map & display filtered points by running the code in Section 1.82.
10. Plot the clusters on a scatter plot by running the code in Section 1.83.
11. Calculate the total severity scores for each cluster by running the code in Section 1.84.
12. Reconstruct Clustered data from files & remove clusters whose total_severity < 100 by running the code in Section 1.85.
13. Batching the data for the Google Distance Matrix API by running the code in Section 1.9.
14. Get distances between Clusters using Google API by running the code in Section 1.91.
15. Follow the steps for using the preprocessed data.
