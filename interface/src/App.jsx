import { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  const handleSearch = async () => {
    if (!query.trim()) return;

    try {
      const response = await fetch(`http://localhost:5000/search?q=${encodeURIComponent(query)}`);
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      console.error("Search error:", error);
    }
  };

  return (
    <div className="app">
      <h1>Search Engine</h1>
      <div className="search-box">
        <input
          type="text"
          placeholder="Search..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
        />
        <button onClick={handleSearch}>Search</button>
      </div>
      <div className="results">
        {results.map((res, index) => (
          <div key={index} className="result">
            <a href={res.url} target="_blank" rel="noopener noreferrer">{res.url}</a>
            <p>Score: {res.score.toFixed(4)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
