import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, ScatterChart, Scatter, Area, AreaChart } from 'recharts';
import * as d3 from 'd3';

const MovieRecommendationDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const d3ChartRef = useRef(null);
  const networkRef = useRef(null);

  // Sample data representing MovieLens dataset insights
  const genreData = [
    { genre: 'Drama', count: 4361, avgRating: 3.7 },
    { genre: 'Comedy', count: 3756, avgRating: 3.4 },
    { genre: 'Thriller', count: 1894, avgRating: 3.6 },
    { genre: 'Action', count: 1828, avgRating: 3.5 },
    { genre: 'Romance', count: 1596, avgRating: 3.5 },
    { genre: 'Adventure', count: 1263, avgRating: 3.6 },
    { genre: 'Crime', count: 1199, avgRating: 3.8 },
    { genre: 'Sci-Fi', count: 980, avgRating: 3.6 },
    { genre: 'Horror', count: 978, avgRating: 3.2 },
    { genre: 'Fantasy', count: 779, avgRating: 3.5 }
  ];

  const ratingDistribution = [
    { rating: '0.5', count: 1370, percentage: 1.4 },
    { rating: '1.0', count: 2811, percentage: 2.8 },
    { rating: '1.5', count: 1791, percentage: 1.8 },
    { rating: '2.0', count: 7551, percentage: 7.6 },
    { rating: '2.5', count: 5550, percentage: 5.6 },
    { rating: '3.0', count: 20047, percentage: 20.1 },
    { rating: '3.5', count: 13136, percentage: 13.2 },
    { rating: '4.0', count: 26818, percentage: 26.9 },
    { rating: '4.5', count: 8551, percentage: 8.6 },
    { rating: '5.0', count: 13211, percentage: 13.3 }
  ];

  const yearlyTrends = [
    { year: 1990, movies: 142, avgRating: 3.6 },
    { year: 1995, movies: 219, avgRating: 3.5 },
    { year: 2000, movies: 396, avgRating: 3.4 },
    { year: 2005, movies: 482, avgRating: 3.3 },
    { year: 2010, movies: 531, avgRating: 3.2 },
    { year: 2015, movies: 612, avgRating: 3.1 },
    { year: 2018, movies: 378, avgRating: 3.0 }
  ];

  const algorithmPerformance = [
    { algorithm: 'User-Based CF', rmse: 0.94, precision: 0.78, recall: 0.65, f1Score: 0.71 },
    { algorithm: 'Item-Based CF', rmse: 0.89, precision: 0.82, recall: 0.69, f1Score: 0.75 },
    { algorithm: 'SVD Matrix', rmse: 0.85, precision: 0.85, recall: 0.72, f1Score: 0.78 },
    { algorithm: 'Content-Based', rmse: 0.91, precision: 0.79, recall: 0.63, f1Score: 0.70 },
    { algorithm: 'Hybrid System', rmse: 0.82, precision: 0.88, recall: 0.76, f1Score: 0.82 }
  ];

  const userActivityData = [
    { userId: 1, totalRatings: 232, avgRating: 4.2, uniqueGenres: 18 },
    { userId: 2, totalRatings: 1056, avgRating: 3.8, uniqueGenres: 19 },
    { userId: 3, totalRatings: 74, avgRating: 3.5, uniqueGenres: 12 },
    { userId: 4, totalRatings: 628, avgRating: 4.1, uniqueGenres: 16 },
    { userId: 5, totalRatings: 349, avgRating: 3.9, uniqueGenres: 15 }
  ];

  // D3.js Heatmap for User-Genre Preferences
  useEffect(() => {
    if (activeTab === 'heatmap' && d3ChartRef.current) {
      createHeatmap();
    }
  }, [activeTab]);

  // D3.js Network Graph for Movie Similarities
  useEffect(() => {
    if (activeTab === 'network' && networkRef.current) {
      createNetworkGraph();
    }
  }, [activeTab]);

  /**
   * createHeatmap function
   * Renders a D3.js heatmap showing user-genre preferences.
   * Visualizes the strength of preference with color intensity.
   */
  const createHeatmap = () => {
    const svg = d3.select(d3ChartRef.current);
    svg.selectAll("*").remove(); // Clear previous chart elements

    const margin = { top: 80, right: 30, bottom: 40, left: 100 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Sample data for heatmap
    const data = [
      { user: 'User 1', genre: 'Action', value: 0.8 },
      { user: 'User 1', genre: 'Comedy', value: 0.6 },
      { user: 'User 1', genre: 'Drama', value: 0.9 },
      { user: 'User 1', genre: 'Sci-Fi', value: 0.7 },
      { user: 'User 2', genre: 'Action', value: 0.5 },
      { user: 'User 2', genre: 'Comedy', value: 0.9 },
      { user: 'User 2', genre: 'Drama', value: 0.4 },
      { user: 'User 2', genre: 'Sci-Fi', value: 0.8 },
      { user: 'User 3', genre: 'Action', value: 0.9 },
      { user: 'User 3', genre: 'Comedy', value: 0.3 },
      { user: 'User 3', genre: 'Drama', value: 0.7 },
      { user: 'User 3', genre: 'Sci-Fi', value: 0.6 }
    ];

    const users = [...new Set(data.map(d => d.user))];
    const genres = [...new Set(data.map(d => d.genre))];

    // X and Y scales for genres and users
    const x = d3.scaleBand().range([0, width]).domain(genres).padding(0.1);
    const y = d3.scaleBand().range([height, 0]).domain(users).padding(0.1);
    // Color scale for preference values
    const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // Add rectangles (tiles) for the heatmap
    g.selectAll(".tile")
      .data(data)
      .enter().append("rect")
      .attr("class", "tile")
      .attr("x", d => x(d.genre))
      .attr("y", d => y(d.user))
      .attr("width", x.bandwidth())
      .attr("height", y.bandwidth())
      .attr("fill", d => colorScale(d.value))
      .style("stroke", "white")
      .style("stroke-width", 2)
      .on("mouseover", function(event, d) {
        // Create and show tooltip on mouseover
        const tooltip = d3.select("body").append("div")
          .attr("class", "tooltip")
          .style("opacity", 0)
          .style("position", "absolute")
          .style("background", "rgba(0,0,0,0.8)")
          .style("color", "white")
          .style("padding", "8px")
          .style("border-radius", "4px")
          .style("font-size", "12px");

        tooltip.transition().duration(200).style("opacity", .9);
        tooltip.html(`${d.user}<br/>${d.genre}: ${d.value}`)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", function() {
        // Remove tooltip on mouseout
        d3.selectAll(".tooltip").remove();
      });

    // Add X and Y axes
    g.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x));
    g.append("g").call(d3.axisLeft(y));

    // Add chart title
    svg.append("text")
      .attr("x", width / 2 + margin.left)
      .attr("y", margin.top / 2)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("User-Genre Preference Heatmap");
  };

  /**
   * createNetworkGraph function
   * Renders a D3.js force-directed network graph showing movie similarities.
   * Nodes represent movies, links represent similarities.
   */
  const createNetworkGraph = () => {
    const svg = d3.select(networkRef.current);
    svg.selectAll("*").remove(); // Clear previous chart elements

    const width = 600;
    const height = 400;

    // Sample data for nodes (movies) and links (similarities)
    const nodes = [
      { id: "Toy Story", group: "Animation", rating: 3.9 },
      { id: "Jumanji", group: "Adventure", rating: 3.2 },
      { id: "Heat", group: "Action", rating: 4.0 },
      { id: "Sabrina", group: "Romance", rating: 3.2 },
      { id: "Sense and Sensibility", group: "Romance", rating: 3.8 },
      { id: "Four Weddings", group: "Comedy", rating: 3.7 },
      { id: "Get Shorty", group: "Comedy", rating: 3.4 }
    ];

    const links = [
      { source: "Toy Story", target: "Jumanji", value: 0.8 },
      { source: "Heat", target: "Sabrina", value: 0.3 },
      { source: "Sense and Sensibility", target: "Four Weddings", value: 0.7 },
      { source: "Four Weddings", target: "Get Shorty", value: 0.9 },
      { source: "Toy Story", target: "Four Weddings", value: 0.5 }
    ];

    // Color scale for movie genres
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Create a force simulation
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(100)) // Link force with distance
      .force("charge", d3.forceManyBody().strength(-300)) // Node repulsion force
      .force("center", d3.forceCenter(width / 2, height / 2)); // Center the graph

    // Add links to the SVG
    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", d => Math.sqrt(d.value) * 3); // Link width based on similarity value

    // Add nodes (circles) to the SVG
    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", d => d.rating * 3) // Node radius based on movie rating
      .attr("fill", d => colorScale(d.group)) // Node color based on genre
      .call(d3.drag() // Enable dragging of nodes
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));

    // Add labels (movie titles) to the nodes
    const labels = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text(d => d.id)
      .style("font-size", "10px")
      .attr("dx", 15)
      .attr("dy", 4);

    // Update link and node positions on each simulation tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      labels
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });

    /**
     * dragstarted function
     * Handles the start of a drag event for a node.
     * @param {object} event - The D3 event object.
     * @param {object} d - The data bound to the dragged node.
     */
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    /**
     * dragged function
     * Handles the dragging of a node.
     * @param {object} event - The D3 event object.
     * @param {object} d - The data bound to the dragged node.
     */
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    /**
     * dragended function
     * Handles the end of a drag event for a node.
     * @param {object} event - The D3 event object.
     * @param {object} d - The data bound to the dragged node.
     */
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  };

  /**
   * CustomTooltip component for Recharts.
   * Renders a customized tooltip for charts, displaying label and payload data.
   */
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-gray-800 text-white p-3 rounded-lg shadow-lg border border-gray-600">
          <p className="font-semibold">{`${label}`}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.dataKey}: ${entry.value}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  // Predefined colors for PieChart cells
  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#8dd1e1', '#d084d0', '#87d068'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 text-white font-inter"> {/* Added font-inter */}
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Movie Recommendation System
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Comprehensive data visualization dashboard showcasing MovieLens dataset insights, 
            algorithm performance, and recommendation system analytics
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="flex flex-wrap justify-center mb-8 space-x-2">
          {[
            { id: 'overview', label: 'Dataset Overview' },
            { id: 'performance', label: 'Algorithm Performance' },
            { id: 'heatmap', label: 'User Preferences' },
            { id: 'network', label: 'Movie Similarity' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all duration-300 ${
                activeTab === tab.id
                  ? 'bg-purple-600 text-white shadow-lg transform scale-105'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content Area */}
        <div className="bg-gray-800 rounded-xl p-8 shadow-2xl">
          {activeTab === 'overview' && (
            <div className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold mb-4">Dataset Overview & Insights</h2>
                <p className="text-gray-300">
                  Analysis of MovieLens dataset showing genre distribution, rating patterns, and temporal trends
                </p>
              </div>

              {/* Genre Distribution */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-gray-700 p-6 rounded-lg">
                  <h3 className="text-xl font-bold mb-4 text-center">Genre Distribution</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={genreData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="genre" tick={{ fontSize: 12 }} angle={-45} textAnchor="end" height={80} />
                      <YAxis />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="count" fill="#8884d8" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-sm text-gray-400 mt-2 text-center">
                    Drama and Comedy dominate the dataset with 4,361 and 3,756 movies respectively
                  </p>
                </div>

                <div className="bg-gray-700 p-6 rounded-lg">
                  <h3 className="text-xl font-bold mb-4 text-center">Rating Distribution</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={ratingDistribution}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="rating" />
                      <YAxis />
                      <Tooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="count" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.6} />
                    </AreaChart>
                  </ResponsiveContainer>
                  <p className="text-sm text-gray-400 mt-2 text-center">
                    Most ratings cluster around 4.0 (26.9%) and 3.0 (20.1%), showing positive bias
                  </p>
                </div>
              </div>

              {/* Yearly Trends */}
              <div className="bg-gray-700 p-6 rounded-lg">
                <h3 className="text-xl font-bold mb-4 text-center">Movie Release Trends Over Time</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={yearlyTrends}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="year" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Bar yAxisId="left" dataKey="movies" fill="#ffc658" name="Movies Released" />
                    <Line yAxisId="right" type="monotone" dataKey="avgRating" stroke="#ff7c7c" strokeWidth={3} name="Avg Rating" />
                  </LineChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-400 mt-2 text-center">
                  Movie production peaked around 2015 with 612 releases, while average ratings show a declining trend
                </p>
              </div>
            </div>
          )}

          {activeTab === 'performance' && (
            <div className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold mb-4">Algorithm Performance Comparison</h2>
                <p className="text-gray-300">
                  Comprehensive evaluation of recommendation algorithms using RMSE, Precision, Recall, and F1-Score
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-gray-700 p-6 rounded-lg">
                  <h3 className="text-xl font-bold mb-4 text-center">RMSE Comparison (Lower is Better)</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={algorithmPerformance} layout="horizontal">
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis type="number" domain={[0, 1]} />
                      <YAxis type="category" dataKey="algorithm" width={100} tick={{ fontSize: 11 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="rmse" fill="#ff7c7c" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  <p className="text-sm text-gray-400 mt-2 text-center">
                    Hybrid System achieves lowest RMSE (0.82), outperforming individual approaches
                  </p>
                </div>

                <div className="bg-gray-700 p-6 rounded-lg">
                  <h3 className="text-xl font-bold mb-4 text-center">Precision vs Recall</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <ScatterChart data={algorithmPerformance}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="recall" name="Recall" />
                      <YAxis dataKey="precision" name="Precision" />
                      <Tooltip cursor={{ stroke: '#8884d8', strokeWidth: 2 }} 
                               content={({ active, payload }) => {
                                 if (active && payload && payload.length) {
                                   const data = payload[0].payload;
                                   return (
                                     <div className="bg-gray-800 text-white p-3 rounded-lg shadow-lg border border-gray-600">
                                       <p className="font-semibold">{data.algorithm}</p>
                                       <p>Precision: {data.precision}</p>
                                       <p>Recall: {data.recall}</p>
                                       <p>F1-Score: {data.f1Score}</p>
                                     </div>
                                   );
                                 }
                                 return null;
                               }} />
                      <Scatter dataKey="precision" fill="#82ca9d" />
                    </ScatterChart>
                  </ResponsiveContainer>
                  <p className="text-sm text-gray-400 mt-2 text-center">
                    Hybrid System shows optimal balance of precision (0.88) and recall (0.76)
                  </p>
                </div>
              </div>

              <div className="bg-gray-700 p-6 rounded-lg">
                <h3 className="text-xl font-bold mb-4 text-center">F1-Score Performance</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={algorithmPerformance}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="algorithm" tick={{ fontSize: 12 }} angle={-15} textAnchor="end" height={80} />
                    <YAxis domain={[0, 1]} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="f1Score" fill="#8dd1e1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-400 mt-2 text-center">
                  Hybrid approach achieves highest F1-Score (0.82), demonstrating superior overall performance
                </p>
              </div>
            </div>
          )}

          {activeTab === 'heatmap' && (
            <div className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold mb-4">User Preference Analysis</h2>
                <p className="text-gray-300">
                  Interactive heatmap showing user preferences across different movie genres
                </p>
              </div>

              <div className="bg-gray-700 p-6 rounded-lg">
                <div className="flex justify-center">
                  <svg ref={d3ChartRef} width="600" height="400"></svg>
                </div>
                <p className="text-sm text-gray-400 mt-4 text-center">
                  Darker colors indicate stronger preferences. Hover over cells for detailed information.
                  This visualization helps identify user clustering and content-based recommendation opportunities.
                </p>
              </div>

              <div className="bg-gray-700 p-6 rounded-lg">
                <h3 className="text-xl font-bold mb-4 text-center">User Activity Patterns</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={userActivityData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="totalRatings" name="Total Ratings" />
                    <YAxis dataKey="avgRating" name="Average Rating" />
                    <Tooltip content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-gray-800 text-white p-3 rounded-lg shadow-lg border border-gray-600">
                            <p className="font-semibold">User {data.userId}</p>
                            <p>Total Ratings: {data.totalRatings}</p>
                            <p>Avg Rating: {data.avgRating}</p>
                            <p>Unique Genres: {data.uniqueGenres}</p>
                          </div>
                        );
                      }
                      return null;
                    }} />
                    <Scatter dataKey="avgRating" fill="#d084d0" />
                  </ScatterChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-400 mt-2 text-center">
                  User activity vs rating behavior - helps identify different user segments for targeted recommendations
                </p>
              </div>
            </div>
          )}

          {activeTab === 'network' && (
            <div className="space-y-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold mb-4">Movie Similarity Network</h2>
                <p className="text-gray-300">
                  Interactive network graph showing relationships between movies based on user ratings and preferences
                </p>
              </div>

              <div className="bg-gray-700 p-6 rounded-lg">
                <div className="flex justify-center">
                  <svg ref={networkRef} width="600" height="400"></svg>
                </div>
                <p className="text-sm text-gray-400 mt-4 text-center">
                  Node size represents movie rating, colors represent genres, and connections show similarity strength.
                  Drag nodes to explore relationships. This network supports item-based collaborative filtering.
                </p>
              </div>

              <div className="bg-gray-700 p-6 rounded-lg">
                <h3 className="text-xl font-bold mb-4 text-center">Genre Preference Distribution</h3>
                <div className="flex justify-center">
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={genreData.slice(0, 7)}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ genre, percentage }) => `${genre} (${((percentage || 0) * 100).toFixed(1)}%)`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="count"
                      >
                        {genreData.slice(0, 7).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip content={<CustomTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-sm text-gray-400 mt-2 text-center">
                  Genre distribution helps understand content diversity and guides content-based recommendations
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Key Insights Summary */}
        <div className="mt-12 bg-gradient-to-r from-purple-800 to-indigo-800 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 text-center">Key Data Insights & Recommendations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-gray-800 p-4 rounded-lg">
              <h3 className="font-bold text-lg mb-2 text-yellow-400">Dataset Characteristics</h3>
              <ul className="text-sm space-y-1 text-gray-300">
                <li>• 9,742 movies across 20 genres</li>
                <li>• 100,836 ratings from 610 users</li>
                <li>• Drama (44.8%) and Comedy (38.6%) dominate</li>
                <li>• Average rating: 3.5/5.0</li>
              </ul>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg">
              <h3 className="font-bold text-lg mb-2 text-green-400">Algorithm Performance</h3>
              <ul className="text-sm space-y-1 text-gray-300">
                <li>• Hybrid System: Best overall (F1: 0.82)</li>
                <li>• SVD Matrix: Lowest RMSE (0.85)</li>
                <li>• Item-based CF: Good precision (0.82)</li>
                <li>• Content-based: Effective for new items/cold start (0.70 F1)</li> {/* Completed line */}
              </ul>
            </div>
            {/* Added new insight section */}
            <div className="bg-gray-800 p-4 rounded-lg">
              <h3 className="font-bold text-lg mb-2 text-blue-400">Scalability Considerations</h3>
              <ul className="text-sm space-y-1 text-gray-300">
                <li>• Data sparsity: Challenge for collaborative filtering</li>
                <li>• Real-time recommendations: Requires optimized algorithms</li>
                <li>• User growth: Demands robust infrastructure</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MovieRecommendationDashboard;
