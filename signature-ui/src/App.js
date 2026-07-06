import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [references, setReferences] = useState([]);
  const [query, setQuery] = useState(null);

  const [result, setResult] = useState("");
  const [distance, setDistance] = useState(null);
  const [threshold, setThreshold] = useState(null);
  const [isBorderline, setIsBorderline] = useState(false);

  // Upload handlers
  const handleRefUpload = (files) => {
    const arr = Array.from(files);
    if (arr.length !== 5) {
      alert("Upload exactly 5 reference signatures");
      return;
    }
    setReferences(arr);
  };

  const handleQueryUpload = (files) => {
    setQuery(files[0]);
  };

  // Submit
  const handleSubmit = async () => {
    if (references.length !== 5 || !query) {
      alert("Upload all required images");
      return;
    }

    const formData = new FormData();
    references.forEach((f) => formData.append("reference", f));
    formData.append("query", query);

    try {
      const API =
      process.env.REACT_APP_API_URL || "http://localhost:5000";

      const res = await axios.post(`${API}/verify`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
     },
     timeout: 120000,
    });

      setResult(res.data.result);
      setDistance(res.data.distance);
      setThreshold(res.data.threshold);
      setIsBorderline(res.data.is_borderline);
    } catch {
      alert("Backend error");
    }
  };

  const getColorClass = () => {
    if (isBorderline) return "orange";
    if (result === "Genuine") return "green";
    if (result === "Forged") return "red";
    return "";
  };

  return (
    <div className="app">
      <h1 className="title">Signature Verification System</h1>

      {/* REFERENCE SECTION */}
      <div className="upload-box">
        <h3>Reference Signatures (5)</h3>

        {references.length === 0 ? (
          <input
            type="file"
            multiple
            onChange={(e) => handleRefUpload(e.target.files)}
          />
        ) : (
          <button
            className="change-btn"
            onClick={() => setReferences([])}
          >
            Change Files
          </button>
        )}

        {/* 🔥 SHOW ONLY 1 IMAGE */}
        {references.length > 0 && (
          <>
            <div className="preview-row">
              <img
                src={URL.createObjectURL(references[0])}
                alt="reference"
              />
            </div>

            {references.length > 1 && (
              <p className="reference-note">
                + {references.length - 1} more similar signatures
              </p>
            )}
          </>
        )}
      </div>

      {/* QUERY SECTION */}
      <div className="upload-box">
        <h3>Query Signature</h3>

        {!query ? (
          <input
            type="file"
            onChange={(e) => handleQueryUpload(e.target.files)}
          />
        ) : (
          <button
            className="change-btn"
            onClick={() => setQuery(null)}
          >
            Change File
          </button>
        )}

        {query && (
          <div className="query-preview">
            <img src={URL.createObjectURL(query)} alt="" />
          </div>
        )}
      </div>

      {/* VERIFY BUTTON */}
      <button className="verify-btn" onClick={handleSubmit}>
        Verify Signature
      </button>

      {/* RESULT */}
      {result && (
        <div className={`result-card ${getColorClass()}`}>
          <h2>{result}</h2>

          {isBorderline && (
            <p className="borderline">⚠ Borderline Case</p>
          )}

          <div className="metrics">
            <div>
              <span>Distance</span>
              <strong>{distance}</strong>
            </div>
            <div>
              <span>Threshold</span>
              <strong>{threshold}</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;