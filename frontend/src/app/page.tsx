"use client";

import React, { useState, useEffect } from "react";
import { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  PieChart, Pie, Cell 
} from "recharts";
import { 
  Upload, Sparkles, RefreshCw, BarChart2, MessageSquare, ShieldCheck, 
  AlertTriangle, Filter, Search, Award, Star, BookOpen, ChevronRight, Layers, ArrowUpRight
} from "lucide-react";

// Types matching Backend schemas
interface Product {
  id: str;
  name: string;
  category: string;
}

interface Theme {
  id: string;
  name: string;
  description: string;
}

interface ReviewAspect {
  aspect: string;
  sentiment: string;
  sentiment_score: number;
  snippet: string;
}

interface Review {
  id: string;
  product_name: string;
  category: string;
  rating: number;
  date: string;
  source: string;
  verified_purchase: boolean;
  helpful_votes: number;
  raw_text: string;
  cleaned_text: string;
  language: string;
  aspects: ReviewAspect[];
}

interface Citation {
  review_id: string;
  product_name: string;
  category: string;
  rating: number;
  date: string;
  snippet: string;
  source: string;
}

interface QAData {
  answer: string;
  citations: Citation[];
  groundedness_score: number;
  confidence_score: number;
  reasoning_summary: string;
  retrieved_count: number;
  is_evidenced: boolean;
}

const BACKEND_URL = "http://localhost:8000/api/v1";

export default function Home() {
  // Filters & Base data
  const [products, setProducts] = useState<Product[]>([]);
  const [themes, setThemes] = useState<Theme[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [totalReviews, setTotalReviews] = useState(0);
  
  // Selection States
  const [selectedProduct, setSelectedProduct] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedRating, setSelectedRating] = useState("");
  const [selectedTheme, setSelectedTheme] = useState("");
  const [trendPeriod, setTrendPeriod] = useState("monthly");

  // Ingestion & Trigger states
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [ingestionStats, setIngestionStats] = useState<any>(null);
  const [discoveringThemes, setDiscoveringThemes] = useState(false);
  const [analyzingSentiment, setAnalyzingSentiment] = useState(false);

  // Trend Data State
  const [trends, setTrends] = useState<any[]>([]);

  // Q&A Chat State
  const [question, setQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [qaResponse, setQaResponse] = useState<QAData | null>(null);

  // Pagination for Review Explorer
  const [page, setPage] = useState(1);
  const itemsPerPage = 6;

  // Notification Banner
  const [message, setMessage] = useState<{ text: string; type: "success" | "error" | "info" } | null>(null);

  // Fetch products, themes, trends and reviews on mount and when filters change
  useEffect(() => {
    fetchMetadata();
  }, []);

  useEffect(() => {
    fetchReviews();
  }, [selectedProduct, selectedCategory, selectedRating, page]);

  useEffect(() => {
    fetchTrends();
  }, [selectedProduct, selectedCategory, selectedTheme, trendPeriod, themes]);

  const showMsg = (text: string, type: "success" | "error" | "info" = "info") => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 5000);
  };

  const fetchMetadata = async () => {
    try {
      const prodRes = await fetch(`${BACKEND_URL}/products`);
      if (prodRes.ok) setProducts(await prodRes.json());
      
      const themeRes = await fetch(`${BACKEND_URL}/themes`);
      if (themeRes.ok) setThemes(await themeRes.json());
    } catch (e) {
      console.error("Error fetching metadata:", e);
    }
  };

  const fetchReviews = async () => {
    try {
      const url = new URL(`${BACKEND_URL}/reviews`);
      if (selectedProduct) url.searchParams.append("product_id", selectedProduct);
      if (selectedCategory) url.searchParams.append("category", selectedCategory);
      if (selectedRating) url.searchParams.append("rating", selectedRating);
      url.searchParams.append("limit", itemsPerPage.toString());
      url.searchParams.append("offset", ((page - 1) * itemsPerPage).toString());

      const res = await fetch(url.toString());
      if (res.ok) {
        const data = await res.json();
        setReviews(data.reviews || []);
        setTotalReviews(data.total || 0);
      }
    } catch (e) {
      console.error("Error fetching reviews:", e);
    }
  };

  const fetchTrends = async () => {
    try {
      const url = new URL(`${BACKEND_URL}/analysis/trends`);
      url.searchParams.append("period_type", trendPeriod);
      if (selectedProduct) url.searchParams.append("product_id", selectedProduct);
      if (selectedCategory) url.searchParams.append("category", selectedCategory);
      if (selectedTheme) url.searchParams.append("theme", selectedTheme);

      const res = await fetch(url.toString());
      if (res.ok) {
        const data = await res.json();
        setTrends(data.trends || []);
      }
    } catch (e) {
      console.error("Error fetching trends:", e);
    }
  };

  // Upload trigger
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", uploadFile);

    try {
      const res = await fetch(`${BACKEND_URL}/reviews/upload`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setIngestionStats(data);
        showMsg(`Successfully ingested ${data.total_inserted} reviews!`, "success");
        setUploadFile(null);
        // Refresh
        fetchMetadata();
        fetchReviews();
        fetchTrends();
      } else {
        const err = await res.json();
        showMsg(`Upload failed: ${err.detail || "Unknown error"}`, "error");
      }
    } catch (e) {
      showMsg("Network error during file upload.", "error");
    } finally {
      setUploading(false);
    }
  };

  // Theme clustering trigger
  const handleThemeDiscovery = async () => {
    setDiscoveringThemes(true);
    try {
      const res = await fetch(`${BACKEND_URL}/analysis/discover-themes?target_num_themes=6`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        showMsg(`Discovered ${data.discovered_count} themes!`, "success");
        fetchMetadata();
        fetchTrends();
      } else {
        showMsg("Theme discovery failed.", "error");
      }
    } catch (e) {
      showMsg("Network error running theme discovery.", "error");
    } finally {
      setDiscoveringThemes(false);
    }
  };

  // ABSA sentiment batch trigger
  const handleAspectSentiment = async () => {
    setAnalyzingSentiment(true);
    try {
      const res = await fetch(`${BACKEND_URL}/analysis/sentiment`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        showMsg(`Sentiment tag completion success! Processed ${data.reviews_processed} reviews.`, "success");
        fetchReviews();
        fetchTrends();
      } else {
        showMsg("Aspect analysis execution failed.", "error");
      }
    } catch (e) {
      showMsg("Network error running sentiment analysis.", "error");
    } finally {
      setAnalyzingSentiment(false);
    }
  };

  // Grounded Query QA trigger
  const handleAskQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setChatLoading(true);
    setQaResponse(null);

    try {
      const payload: any = { question };
      if (selectedProduct) payload.product_id = selectedProduct;
      if (selectedCategory) payload.category = selectedCategory;
      if (selectedRating) payload.rating_filter = [parseInt(selectedRating)];

      const res = await fetch(`${BACKEND_URL}/qa/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        setQaResponse(data);
      } else {
        showMsg("Failed to get answer from QA engine.", "error");
      }
    } catch (e) {
      showMsg("Network error sending question.", "error");
    } finally {
      setChatLoading(false);
    }
  };

  const getAvgRating = () => {
    if (reviews.length === 0) return 0;
    const sum = reviews.reduce((acc, r) => acc + r.rating, 0);
    return (sum / reviews.length).toFixed(1);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans p-6 selection:bg-teal-500 selection:text-slate-900">
      
      {/* Banner message */}
      {message && (
        <div className={`fixed top-4 right-4 z-50 px-4 py-3 rounded-lg border shadow-xl flex items-center gap-3 animate-bounce transition-all ${
          message.type === "success" ? "bg-teal-950/80 border-teal-500 text-teal-300" :
          message.type === "error" ? "bg-rose-950/80 border-rose-500 text-rose-300" :
          "bg-slate-900/80 border-slate-700 text-slate-300"
        }`}>
          <Layers className="w-5 h-5" />
          <span className="text-sm font-semibold">{message.text}</span>
        </div>
      )}

      {/* Top Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-5 mb-8 gap-4">
        <div>
          <div className="flex items-center gap-3">
            <span className="px-3 py-1 bg-gradient-to-r from-teal-500 to-indigo-600 rounded-lg text-xs font-bold tracking-widest text-slate-900 uppercase">Havells R&D</span>
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-slate-50 to-slate-400 bg-clip-text text-transparent">
              Customer Voice Intelligence
            </h1>
          </div>
          <p className="text-slate-400 text-sm mt-1">Enterprise Grounded AI Insights Engine for Product Managers</p>
        </div>

        {/* Global Filter Bar */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Category Filter */}
          <div className="flex items-center gap-2 bg-slate-900/70 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300">
            <Filter className="w-3.5 h-3.5 text-teal-400" />
            <select 
              value={selectedCategory} 
              onChange={(e) => { setSelectedCategory(e.target.value); setPage(1); }}
              className="bg-transparent border-none focus:outline-none cursor-pointer"
            >
              <option value="" className="bg-slate-900">All Categories</option>
              <option value="Fans" className="bg-slate-900">Fans</option>
              <option value="Water Heaters" className="bg-slate-900">Water Heaters</option>
              <option value="Air Purifiers" className="bg-slate-900">Air Purifiers</option>
            </select>
          </div>

          {/* Product Filter */}
          <div className="flex items-center gap-2 bg-slate-900/70 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300">
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            <select 
              value={selectedProduct} 
              onChange={(e) => { setSelectedProduct(e.target.value); setPage(1); }}
              className="bg-transparent border-none focus:outline-none cursor-pointer max-w-[200px]"
            >
              <option value="" className="bg-slate-900">All Products</option>
              {products.map((p) => (
                <option key={p.id} value={p.id} className="bg-slate-900">{p.name}</option>
              ))}
            </select>
          </div>

          {/* Rating Filter */}
          <div className="flex items-center gap-2 bg-slate-900/70 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300">
            <Star className="w-3.5 h-3.5 text-amber-400" />
            <select 
              value={selectedRating} 
              onChange={(e) => { setSelectedRating(e.target.value); setPage(1); }}
              className="bg-transparent border-none focus:outline-none cursor-pointer"
            >
              <option value="" className="bg-slate-900">All Ratings</option>
              <option value="5" className="bg-slate-900">5 Stars</option>
              <option value="4" className="bg-slate-900">4 Stars</option>
              <option value="3" className="bg-slate-900">3 Stars</option>
              <option value="2" className="bg-slate-900">2 Stars</option>
              <option value="1" className="bg-slate-900">1 Star</option>
            </select>
          </div>

          {/* Clear Filters Button */}
          {(selectedProduct || selectedCategory || selectedRating || selectedTheme) && (
            <button 
              onClick={() => {
                setSelectedProduct("");
                setSelectedCategory("");
                setSelectedRating("");
                setSelectedTheme("");
                setPage(1);
              }}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs font-semibold rounded-lg flex items-center gap-1.5 cursor-pointer text-slate-300"
            >
              <RefreshCw className="w-3 h-3" /> Clear
            </button>
          )}
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* LEFT COLUMN: Data Pipeline Operations & Stats */}
        <div className="space-y-8 lg:col-span-1">
          
          {/* Card 1: Key Performance Indicators */}
          <div className="bg-slate-900/50 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
              <Award className="w-4 h-4 text-teal-400" /> Intelligence Summary
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/50">
                <span className="text-xs text-slate-500 font-semibold block">Total Reviews</span>
                <span className="text-2xl font-black text-slate-100">{totalReviews}</span>
              </div>
              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/50">
                <span className="text-xs text-slate-500 font-semibold block">Average Rating</span>
                <span className="text-2xl font-black text-amber-400 flex items-center gap-1">
                  {getAvgRating()} <Star className="w-5 h-5 fill-current inline" />
                </span>
              </div>
              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/50 col-span-2">
                <span className="text-xs text-slate-500 font-semibold block">Discovered Themes</span>
                <span className="text-lg font-bold text-slate-300 mt-1">{themes.length || 0} active topics</span>
              </div>
            </div>
          </div>

          {/* Card 2: Operations Panel (Ingestion & Triggering) */}
          <div className="bg-slate-900/50 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md space-y-6">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" /> Pipeline Operations
            </h2>

            {/* Ingestion upload */}
            <form onSubmit={handleUpload} className="space-y-3">
              <label className="block text-xs font-semibold text-slate-400">Upload Reviews Dataset (CSV/JSON)</label>
              <div className="flex gap-2">
                <input 
                  type="file" 
                  accept=".csv,.json"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="block w-full text-xs text-slate-400 file:mr-4 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-850 file:text-slate-300 file:cursor-pointer bg-slate-950/50 border border-slate-850 rounded-lg p-1"
                />
                <button 
                  type="submit" 
                  disabled={uploading || !uploadFile}
                  className="bg-indigo-650 hover:bg-indigo-600 disabled:bg-slate-800 disabled:text-slate-500 px-3 rounded-lg text-xs font-semibold cursor-pointer text-slate-100 transition-colors flex items-center gap-1.5"
                >
                  {uploading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}
                  Ingest
                </button>
              </div>
              {ingestionStats && (
                <div className="text-[11px] text-teal-400 bg-teal-950/30 border border-teal-900/50 p-2.5 rounded-lg space-y-0.5">
                  <span className="font-bold block">Upload Success Metrics:</span>
                  <span>Total Parsed: {ingestionStats.total_received}</span>
                  <span className="block">Inserted: {ingestionStats.total_inserted} | Skipped: {ingestionStats.total_duplicates_skipped}</span>
                </div>
              )}
            </form>

            <div className="border-t border-slate-850 pt-4 space-y-3">
              <span className="block text-xs font-semibold text-slate-400">Analysis Agents Trigger</span>
              
              <div className="grid grid-cols-2 gap-3">
                <button 
                  onClick={handleThemeDiscovery}
                  disabled={discoveringThemes}
                  className="bg-slate-950 hover:bg-slate-900 border border-slate-800 disabled:opacity-50 text-[11px] py-2 px-3 rounded-xl font-bold cursor-pointer transition-all flex items-center justify-center gap-1.5"
                >
                  {discoveringThemes ? <RefreshCw className="w-3.5 h-3.5 animate-spin text-teal-400" /> : <Layers className="w-3.5 h-3.5 text-teal-400" />}
                  Cluster Themes
                </button>

                <button 
                  onClick={handleAspectSentiment}
                  disabled={analyzingSentiment}
                  className="bg-slate-950 hover:bg-slate-900 border border-slate-800 disabled:opacity-50 text-[11px] py-2 px-3 rounded-xl font-bold cursor-pointer transition-all flex items-center justify-center gap-1.5"
                >
                  {analyzingSentiment ? <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-400" /> : <Sparkles className="w-3.5 h-3.5 text-indigo-400" />}
                  Extract Sentiment
                </button>
              </div>
            </div>
          </div>

          {/* Card 3: Theme Cloud */}
          <div className="bg-slate-900/50 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" /> Auto Discovered Themes
            </h2>
            <div className="flex flex-wrap gap-2.5 mt-2">
              {themes.length === 0 ? (
                <span className="text-xs text-slate-500">No themes discovered yet. Click &quot;Cluster Themes&quot; to discover topics.</span>
              ) : (
                themes.map((t) => (
                  <button 
                    key={t.id} 
                    onClick={() => setSelectedTheme(selectedTheme === t.name ? "" : t.name)}
                    className={`text-[11px] font-bold py-1.5 px-3 rounded-full border transition-all cursor-pointer ${
                      selectedTheme === t.name 
                        ? "bg-teal-500/10 border-teal-500 text-teal-300"
                        : "bg-slate-950/60 border-slate-850 hover:border-slate-700 text-slate-400"
                    }`}
                  >
                    #{t.name}
                  </button>
                ))
              )}
            </div>
          </div>

        </div>

        {/* MIDDLE COLUMN: Grounded QA Workspace & Citations */}
        <div className="lg:col-span-2 space-y-8">
          
          {/* Card 4: Grounded Agent Q&A Workspace */}
          <div className="bg-slate-900/50 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-indigo-400" /> Grounded QA Workspace
            </h2>

            <form onSubmit={handleAskQuestion} className="space-y-4">
              <div className="relative">
                <input 
                  type="text" 
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question (e.g., 'Compare water heaters and mixer grinders' or 'What are the motor noise issues in ceiling fans?')"
                  className="w-full bg-slate-950 border border-slate-850 rounded-xl py-3 pl-4 pr-12 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-teal-500 transition-colors"
                />
                <button 
                  type="submit"
                  disabled={chatLoading || !question.trim()}
                  className="absolute right-2 top-2 p-1.5 bg-gradient-to-r from-teal-500 to-indigo-600 rounded-lg text-slate-900 hover:opacity-90 disabled:opacity-50 transition-opacity cursor-pointer"
                >
                  {chatLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ChevronRight className="w-4 h-4" />}
                </button>
              </div>
            </form>

            {/* QA Output Display */}
            {qaResponse && (
              <div className="mt-6 space-y-6 animate-fadeIn">
                
                {/* Metrics header */}
                <div className="flex flex-wrap gap-4 border-b border-slate-850 pb-4">
                  
                  {/* Groundedness Badge */}
                  <div className={`px-3 py-1.5 rounded-lg border text-xs font-bold flex items-center gap-1.5 ${
                    qaResponse.groundedness_score >= 0.8 
                      ? "bg-teal-950/40 border-teal-500 text-teal-400"
                      : "bg-rose-950/40 border-rose-500 text-rose-400"
                  }`}>
                    {qaResponse.groundedness_score >= 0.8 ? <ShieldCheck className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
                    Groundedness: {(qaResponse.groundedness_score * 100).toFixed(0)}%
                  </div>

                  {/* Confidence Badge */}
                  <div className="px-3 py-1.5 bg-indigo-950/40 border border-indigo-800 text-indigo-400 rounded-lg text-xs font-bold flex items-center gap-1.5">
                    <Star className="w-4 h-4" />
                    Confidence: {(qaResponse.confidence_score * 100).toFixed(0)}%
                  </div>

                  {/* retrieved docs */}
                  <div className="px-3 py-1.5 bg-slate-950/50 border border-slate-850 text-slate-400 rounded-lg text-xs font-bold flex items-center gap-1.5">
                    <BookOpen className="w-4 h-4" />
                    Context Documents: {qaResponse.retrieved_count}
                  </div>
                </div>

                {/* Main answer text */}
                <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-5 space-y-3">
                  <span className="text-[11px] font-bold text-slate-500 uppercase tracking-widest block">Agent Generated Response</span>
                  <div className="text-slate-200 text-sm leading-relaxed whitespace-pre-line">
                    {qaResponse.answer}
                  </div>
                </div>

                {/* Audit detail */}
                <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-850 text-xs">
                  <span className="font-bold text-slate-400 block mb-1">Explainability reasoning summary:</span>
                  <p className="text-slate-400 leading-normal">{qaResponse.reasoning_summary}</p>
                </div>

                {/* Citations panel */}
                {qaResponse.citations.length > 0 && (
                  <div className="space-y-3">
                    <span className="text-[11px] font-bold text-slate-500 uppercase tracking-widest block">Evidence & Grounding Citations</span>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {qaResponse.citations.map((cit, idx) => (
                        <div key={idx} className="bg-slate-950/80 border border-slate-850 p-4 rounded-xl space-y-2">
                          <div className="flex items-center justify-between border-b border-slate-900 pb-1.5">
                            <span className="text-xs font-bold text-teal-400 block truncate max-w-[150px]">{cit.product_name}</span>
                            <span className="text-[10px] text-slate-500">{new Date(cit.date).toLocaleDateString()}</span>
                          </div>
                          <p className="text-slate-400 text-xs italic leading-relaxed">&ldquo;{cit.snippet}&rdquo;</p>
                          <div className="flex items-center justify-between text-[10px] text-slate-500 pt-1">
                            <span>Review: [{cit.review_id.slice(0, 8)}...]</span>
                            <span className="bg-slate-900 text-slate-400 px-2 py-0.5 rounded border border-slate-800 flex items-center gap-0.5">
                              {cit.rating} <Star className="w-2.5 h-2.5 fill-current text-amber-500" />
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              </div>
            )}

            {chatLoading && (
              <div className="mt-8 flex flex-col items-center justify-center gap-3 py-10">
                <RefreshCw className="w-8 h-8 text-teal-400 animate-spin" />
                <span className="text-xs text-slate-400">Verifying evidence and drafting answer...</span>
              </div>
            )}
          </div>

          {/* Card 5: Trend Graphs Visualizer */}
          <div className="bg-slate-900/50 border border-slate-800/80 rounded-2xl p-6 backdrop-blur-md">
            <div className="flex items-center justify-between border-b border-slate-850 pb-4 mb-6">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-emerald-400" /> Sentiment & Theme Trends
              </h2>
              <div className="flex items-center gap-2 bg-slate-950/70 border border-slate-850 rounded-lg px-2 py-1 text-xs">
                <select 
                  value={trendPeriod} 
                  onChange={(e) => setTrendPeriod(e.target.value)}
                  className="bg-transparent border-none focus:outline-none cursor-pointer text-slate-300"
                >
                  <option value="weekly" className="bg-slate-900">Weekly</option>
                  <option value="monthly" className="bg-slate-900">Monthly</option>
                  <option value="quarterly" className="bg-slate-900">Quarterly</option>
                </select>
              </div>
            </div>

            <div className="h-64">
              {trends.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <span className="text-xs text-slate-500">No trend data available. Ingest reviews and run aspect mapping to generate graphs.</span>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trends}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="period" stroke="#94a3b8" fontSize={11} />
                    <YAxis yAxisId="left" stroke="#94a3b8" fontSize={11} />
                    <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" fontSize={11} />
                    <Tooltip contentStyle={{ backgroundColor: "#020617", borderColor: "#334155" }} labelClassName="text-slate-300 text-xs font-bold" />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line yAxisId="left" type="monotone" dataKey="total_reviews" name="Review Count" stroke="#3b82f6" strokeWidth={2.5} activeDot={{ r: 8 }} />
                    <Line yAxisId="right" type="monotone" dataKey="average_rating" name="Avg Rating" stroke="#f59e0b" strokeWidth={2.5} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

        </div>

      </div>

      {/* SECTION: Review Explorer Table */}
      <section className="bg-slate-900/50 border border-slate-800/80 rounded-2xl p-6 mt-8 backdrop-blur-md">
        <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-850 pb-4 mb-4 gap-4">
          <div>
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-teal-400" /> Review Explorer
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">Showing {reviews.length} of {totalReviews} reviews</p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500 uppercase tracking-wider font-semibold">
                <th className="py-3 px-4">Product / Category</th>
                <th className="py-3 px-4">Rating</th>
                <th className="py-3 px-4">Feedback</th>
                <th className="py-3 px-4">Aspect Sentiments</th>
                <th className="py-3 px-4">Date / Source</th>
              </tr>
            </thead>
            <tbody>
              {reviews.length === 0 ? (
                <tr>
                  <td colSpan={5} className="py-10 text-center text-slate-500">No reviews matching the filters were found.</td>
                </tr>
              ) : (
                reviews.map((r) => (
                  <tr key={r.id} className="border-b border-slate-850 hover:bg-slate-900/40 transition-colors">
                    <td className="py-4 px-4 max-w-[200px]">
                      <span className="font-bold text-slate-300 block truncate">{r.product_name}</span>
                      <span className="text-[10px] text-slate-500">{r.category}</span>
                    </td>
                    <td className="py-4 px-4">
                      <span className="bg-slate-950 border border-slate-800 px-2 py-1 rounded text-amber-400 font-bold flex items-center gap-1 w-fit">
                        {r.rating} <Star className="w-3.5 h-3.5 fill-current" />
                      </span>
                    </td>
                    <td className="py-4 px-4 max-w-sm">
                      <p className="text-slate-300 line-clamp-3 leading-normal">{r.cleaned_text || r.raw_text}</p>
                      {r.language !== "en" && (
                        <span className="text-[9px] text-indigo-400 font-bold mt-1 block">Translated from: {r.language.toUpperCase()}</span>
                      )}
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex flex-wrap gap-1.5 max-w-[250px]">
                        {r.aspects.length === 0 ? (
                          <span className="text-[10px] text-slate-650 italic">No aspects mapped</span>
                        ) : (
                          r.aspects.map((asp, idx) => (
                            <span 
                              key={idx} 
                              className={`text-[9px] font-bold px-2 py-0.5 rounded border ${
                                asp.sentiment === "Positive" ? "bg-emerald-950/40 border-emerald-500/50 text-emerald-400" :
                                asp.sentiment === "Negative" ? "bg-rose-950/40 border-rose-500/50 text-rose-400" :
                                "bg-slate-950 border-slate-800 text-slate-400"
                              }`}
                            >
                              {asp.aspect} ({asp.sentiment === "Positive" ? "+" : asp.sentiment === "Negative" ? "-" : "~"})
                            </span>
                          ))
                        )}
                      </div>
                    </td>
                    <td className="py-4 px-4 text-slate-400">
                      <span>{new Date(r.date).toLocaleDateString()}</span>
                      <span className="text-[10px] text-slate-500 block">{r.source || "Direct"}</span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination controls */}
        {totalReviews > itemsPerPage && (
          <div className="flex items-center justify-between border-t border-slate-850 pt-4 mt-4">
            <button 
              disabled={page === 1}
              onClick={() => setPage(page - 1)}
              className="px-3.5 py-1.5 bg-slate-950 hover:bg-slate-900 border border-slate-850 disabled:opacity-50 text-xs font-bold rounded-lg cursor-pointer transition-colors"
            >
              Previous
            </button>
            <span className="text-xs text-slate-500">Page {page} of {Math.ceil(totalReviews / itemsPerPage)}</span>
            <button 
              disabled={page >= Math.ceil(totalReviews / itemsPerPage)}
              onClick={() => setPage(page + 1)}
              className="px-3.5 py-1.5 bg-slate-950 hover:bg-slate-900 border border-slate-850 disabled:opacity-50 text-xs font-bold rounded-lg cursor-pointer transition-colors"
            >
              Next
            </button>
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className="mt-16 text-center text-[10px] text-slate-650 border-t border-slate-900 pt-6">
        Havells Customer Voice Intelligence Agent System &bull; Grounded RAG &bull; Dynamic Clustering &bull; Aspect ABSA
      </footer>

    </div>
  );
}
