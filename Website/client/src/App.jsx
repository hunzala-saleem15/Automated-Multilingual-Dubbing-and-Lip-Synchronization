import React, { useRef, useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate, useLocation } from 'react-router-dom';
import axios from "axios";

// Icons (Sabhi icons ko ek hi jagah combine kar diya gaya hai)
import { 
  CloudArrowUpIcon, 
  XMarkIcon as XIcon, 
  Bars3Icon as MenuIcon, 
  UserCircleIcon,
  VideoCameraIcon, 
  ClockIcon, 
  CreditCardIcon, 
  ArrowUpIcon,
  ArrowDownTrayIcon,
  CameraIcon,
  BookmarkIcon,
  ExclamationTriangleIcon,
  SparklesIcon,
  PlayIcon
} from '@heroicons/react/24/outline';

import { CheckIcon, MoonIcon, SunIcon } from '@heroicons/react/24/solid';

// Recharts
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';

// Assets
import avatar from "./assets/images/avatar.png";
import './index.css';
import VisaImg from "./assets/images/Visa.png";
import myVideo from "./assets/video.mp4";


// Navbar 
const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  
  const [darkMode, setDarkMode] = useState(() => JSON.parse(localStorage.getItem("darkMode")) || false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userPhoto, setUserPhoto] = useState(null);

  const BASE_URL = "http://127.0.0.1:8000";

  useEffect(() => {
    if (darkMode) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
    localStorage.setItem("darkMode", JSON.stringify(darkMode));
  }, [darkMode]);

  useEffect(() => {
    const checkUser = () => {
      const data = sessionStorage.getItem("userData");
      if (data) {
        const parsedUser = JSON.parse(data);
        setIsLoggedIn(true);
        if (parsedUser.profile_photo) {
          setUserPhoto(`${BASE_URL}/${parsedUser.profile_photo}`);
        } else {
          setUserPhoto(null);
        }
      } else {
        setIsLoggedIn(false);
        setUserPhoto(null);
      }
    };

    checkUser();
    setMobileMenuOpen(false);

    window.addEventListener('storage', checkUser);
    return () => window.removeEventListener('storage', checkUser);
  }, [location.pathname]);

  const handleLogout = () => {
    sessionStorage.clear();
    localStorage.removeItem("token");
    setIsLoggedIn(false);
    navigate("/login");
  };

  const links = [
    { path: "/", label: "Home" },
    { path: "/pricing", label: "Pricing" },
    { path: "/about", label: "About" },
    { path: "/contact", label: "Contact" },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-zinc-950/80 backdrop-blur-md border-b border-zinc-200/80 dark:border-zinc-800/80 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex justify-between items-center">
        
        {/* LOGO */}
        <Link to="/" className="font-bold text-xl tracking-tight text-zinc-900 dark:text-white flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-teal-500 inline-block"></span>
          SyncDub AI
        </Link>

        {/* DESKTOP LINKS */}
        <div className="hidden md:flex items-center space-x-8">
          <div className="flex items-center space-x-6">
            {links.map((link) => (
              <Link 
                key={link.path} 
                to={link.path} 
                className={`text-sm font-medium transition-colors ${
                  location.pathname === link.path 
                    ? "text-zinc-900 dark:text-white" 
                    : "text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white"
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>

          <div className="w-px h-4 bg-zinc-200 dark:bg-zinc-800"></div>

          {isLoggedIn ? (
            <div className="flex items-center space-x-4">
              <Link 
                to="/upload" 
                className="text-xs font-semibold bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 px-3.5 py-2 rounded-md hover:bg-zinc-800 dark:hover:bg-white transition-colors"
              >
                + Upload
              </Link>
              <Link to="/profile">
                {userPhoto ? (
                  <img src={userPhoto} alt="Profile" className="h-8 w-8 rounded-full border border-zinc-200 dark:border-zinc-800 object-cover" />
                ) : (
                  <UserCircleIcon className="h-8 w-8 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors" />
                )}
              </Link>
              <button onClick={handleLogout} className="text-xs font-medium text-zinc-500 hover:text-red-600 transition-colors">
                Logout
              </button>
            </div>
          ) : (
            <div className="flex items-center space-x-3">
              <Link to="/login" className="text-sm font-medium text-zinc-600 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-white transition-colors">
                Log in
              </Link>
              <Link 
                to="/signup" 
                className="text-sm font-medium bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 px-4 py-2 rounded-md hover:bg-zinc-800 dark:hover:bg-zinc-100 transition-colors"
              >
                Sign Up
              </Link>
            </div>
          )}

          <button 
            onClick={() => setDarkMode(!darkMode)} 
            className="p-1.5 rounded-md border border-zinc-200 dark:border-zinc-800 text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white transition-colors"
            aria-label="Toggle Theme"
          >
            {darkMode ? <SunIcon className="h-4 w-4" /> : <MoonIcon className="h-4 w-4" />}
          </button>
        </div>

        {/* MOBILE BUTTONS */}
        <div className="flex md:hidden items-center space-x-2">
          <button 
            onClick={() => setDarkMode(!darkMode)} 
            className="p-2 rounded-md text-zinc-500 dark:text-zinc-400"
          >
            {darkMode ? <SunIcon className="h-5 w-5" /> : <MoonIcon className="h-5 w-5" />}
          </button>
          
          <button 
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 text-zinc-700 dark:text-zinc-200"
          >
            {mobileMenuOpen ? <XIcon className="h-6 w-6" /> : <MenuIcon className="h-6 w-6" />}
          </button>
        </div>
      </div>

      {/* MOBILE MENU */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-white dark:bg-zinc-950 border-b border-zinc-200 dark:border-zinc-800 px-4 pt-2 pb-6 space-y-1">
          {links.map((link) => (
            <Link 
              key={link.path} 
              to={link.path} 
              className="block py-2 px-3 rounded-md text-sm font-medium text-zinc-700 dark:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-900"
            >
              {link.label}
            </Link>
          ))}
          
          <div className="h-px bg-zinc-200 dark:bg-zinc-800 my-3"></div>

          {isLoggedIn ? (
            <div className="space-y-2 pt-1">
              <Link to="/profile" className="flex items-center space-x-3 py-2 px-3 rounded-md hover:bg-zinc-100 dark:hover:bg-zinc-900">
                {userPhoto ? (
                  <img src={userPhoto} className="h-8 w-8 rounded-full border border-zinc-200 dark:border-zinc-800" alt="" />
                ) : (
                  <UserCircleIcon className="h-8 w-8 text-zinc-400" />
                )}
                <span className="text-sm font-medium text-zinc-900 dark:text-white">My Profile</span>
              </Link>
              <Link to="/upload" className="block text-center py-2.5 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 rounded-md text-sm font-semibold">
                + Create New Dub
              </Link>
              <button onClick={handleLogout} className="w-full text-left py-2 px-3 text-sm font-medium text-red-600">
                Logout
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2 pt-2">
              <Link to="/login" className="py-2 text-center text-sm font-medium border border-zinc-200 dark:border-zinc-800 rounded-md text-zinc-700 dark:text-zinc-200">
                Log in
              </Link>
              <Link to="/signup" className="py-2 text-center text-sm font-medium bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 rounded-md">
                Sign Up
              </Link>
            </div>
          )}
        </div>
      )}
    </nav>
  );
};

// Home 
const Home = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    const user = sessionStorage.getItem("userData");
    setIsLoggedIn(!!user);
  }, []);

  return (
    <div className="relative min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 flex flex-col justify-center px-4 sm:px-6 lg:px-8 py-24 border-b border-zinc-200 dark:border-zinc-800">
      
      {/* Subtle Grid Background instead of loud blurred blobs */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none"></div>

      <div className="relative z-10 max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 items-center gap-12 lg:gap-16">
        
        {/* Left Content Section */}
        <div className="text-center lg:text-left space-y-6">
          
          <div className="inline-flex items-center px-3 py-1 rounded-full border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 shadow-sm">
            <span className="flex h-2 w-2 bg-teal-500 rounded-full mr-2"></span>
            <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400 tracking-wide uppercase">
              SyncDub AI v2.0 Released
            </span>
          </div>
          
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-zinc-900 dark:text-white leading-[1.1]">
            Reach global audiences <br />
            <span className="text-zinc-500 dark:text-zinc-400">
              in any language.
            </span>
          </h1>
          
          <p className="text-zinc-600 dark:text-zinc-400 text-base sm:text-lg leading-relaxed max-w-xl mx-auto lg:mx-0">
            Automatically dub your videos with precise lip-syncing and natural, human-like voice synthesis powered by state-of-the-art AI.
          </p>
          
          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 pt-2 justify-center lg:justify-start">
            {isLoggedIn ? (
              <Link
                to="/dashboard"
                className="w-full sm:w-auto flex items-center justify-center px-6 py-3 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 rounded-md font-medium text-sm hover:bg-zinc-800 dark:hover:bg-zinc-100 transition-colors shadow-sm"
              >
                Go to Dashboard
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </Link>
            ) : (
              <>
                <Link
                  to="/upload"
                  className="w-full sm:w-auto flex items-center justify-center px-6 py-3 bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 rounded-md font-medium text-sm hover:bg-zinc-800 dark:hover:bg-zinc-100 transition-colors shadow-sm"
                >
                  Get Started Free
                </Link>
                <Link
                  to="/signup"
                  className="w-full sm:w-auto flex items-center justify-center px-6 py-3 border border-zinc-300 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 rounded-md font-medium text-sm hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors"
                >
                  View Demo
                </Link>
              </>
            )}
          </div>
        </div>
        
        {/* Right Video Preview */}
        <div className="flex justify-center lg:justify-end">
          <div className="w-full max-w-2xl rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-2 shadow-xl">
            <div className="rounded-md overflow-hidden bg-zinc-950 aspect-video relative">
              <video
                src={myVideo}
                controls 
                autoPlay 
                muted 
                loop 
                playsInline
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

// Dashboard
const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [videoData, setVideoData] = useState({ total: 0, recent: [] });
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const token = localStorage.getItem("token");
        const res = await axios.get("http://127.0.0.1:8000/auth/me", {
          headers: { Authorization: `Bearer ${token}` }
        });

        const userData = res.data;
        const videos = userData.saved_videos || [];

        setUser(userData);
        setVideoData({
          total: videos.length,
          recent: [...videos].reverse().slice(0, 3) 
        });

        generateChartStats(videos);
      } catch (err) {
        console.error("Dashboard error:", err);
      } finally {
        setLoading(false);
      }
    };

    const generateChartStats = (videos) => {
      const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      const last7Days = [];
      for (let i = 6; i >= 0; i--) {
        const d = new Date();
        d.setDate(d.getDate() - i);
        last7Days.push({ day: days[d.getDay()], dateKey: d.toDateString(), mins: 0 });
      }
      videos.forEach(video => {
        const rawDate = video.saved_at?.$date || video.saved_at;
        const vDateString = new Date(rawDate).toDateString();
        const dayMatch = last7Days.find(d => d.dateKey === vDateString);
        if (dayMatch) dayMatch.mins += 10;
      });
      setChartData(last7Days);
    };

    fetchDashboardData();
  }, []);

  const isPremium = user?.plan === "Premium" || user?.plan === "Enterprise";

  const statsCards = [
    { 
      label: "Videos Dubbed", 
      value: videoData.total, 
      total: isPremium ? "Unlimited" : "5", 
      icon: <VideoCameraIcon className="w-5 h-5 text-zinc-700 dark:text-zinc-300"/>, 
    },
    { 
      label: "Current Plan", 
      value: user?.plan?.toUpperCase() || "FREE", 
      total: "Active", 
      icon: <CreditCardIcon className="w-5 h-5 text-zinc-700 dark:text-zinc-300"/>, 
    },
    { 
      label: "Storage Policy", 
      value: isPremium ? "Permanent" : "24h Retain", 
      total: isPremium ? "Unlimited" : "Auto-cleared", 
      icon: <ClockIcon className="w-5 h-5 text-zinc-700 dark:text-zinc-300"/>, 
    },
  ];

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-zinc-50 dark:bg-zinc-950">
      <div className="animate-spin rounded-full h-8 w-8 border-2 border-zinc-900 dark:border-white border-t-transparent"></div>
    </div>
  );

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 pt-24 pb-12 px-4 sm:px-6 lg:px-8 border-b border-zinc-200 dark:border-zinc-800">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header Section */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-zinc-200 dark:border-zinc-800">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-zinc-900 dark:text-white">
              Welcome back, {user?.first_name || user?.username}
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
              {isPremium ? "You have Pro access to all models and features." : `You have created ${videoData.total} dubbed projects on the free tier.`}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {!isPremium && (
              <Link 
                to="/pricing" 
                className="flex items-center gap-2 text-xs font-semibold border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-zinc-800 dark:text-zinc-200 px-4 py-2.5 rounded-md hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors shadow-sm"
              >
                <SparklesIcon className="w-4 h-4 text-amber-500" />
                Upgrade
              </Link>
            )}
            
            <Link 
              to="/upload" 
              className="text-xs font-semibold bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 px-4 py-2.5 rounded-md hover:bg-zinc-800 dark:hover:bg-zinc-100 transition-colors shadow-sm"
            >
              + New Project
            </Link>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {statsCards.map((stat, i) => (
            <div 
              key={i} 
              className="bg-white dark:bg-zinc-900 p-5 rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm flex items-center justify-between"
            >
              <div className="space-y-1">
                <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">{stat.label}</p>
                <div className="flex items-baseline gap-2">
                  <h3 className="text-xl font-semibold text-zinc-900 dark:text-white tracking-tight">{stat.value}</h3>
                  <span className="text-xs text-zinc-400">/ {stat.total}</span>
                </div>
              </div>
              <div className="p-2.5 rounded-md border border-zinc-100 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950">
                {stat.icon}
              </div>
            </div>
          ))}
        </div>

        {/* Analytics & Recent Projects */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Chart Card */}
          <div className="lg:col-span-2 bg-white dark:bg-zinc-900 p-6 rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-base font-semibold text-zinc-900 dark:text-white">Usage Analytics</h2>
                <p className="text-xs text-zinc-500">Dubbing duration output in minutes</p>
              </div>
            </div>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorMins" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#71717a" stopOpacity={0.2}/>
                      <stop offset="95%" stopColor="#71717a" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#27272a" opacity={0.15} />
                  <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{fill: '#a1a1aa', fontSize: 12}} />
                  <YAxis axisLine={false} tickLine={false} tick={{fill: '#a1a1aa', fontSize: 12}} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#18181b', 
                      borderColor: '#27272a', 
                      borderRadius: '6px',
                      color: '#fff',
                      fontSize: '12px'
                    }}
                  />
                  <Area type="monotone" dataKey="mins" stroke="#71717a" strokeWidth={2} fill="url(#colorMins)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Recent Videos Card */}
          <div className="bg-white dark:bg-zinc-900 p-6 rounded-lg border border-zinc-200 dark:border-zinc-800 shadow-sm flex flex-col justify-between">
            <div>
              <h2 className="text-base font-semibold text-zinc-900 dark:text-white mb-4">Recent Saved Videos</h2>
              <div className="space-y-3">
                {videoData.recent.length > 0 ? (
                  videoData.recent.map((video, idx) => (
                    <div 
                      key={idx} 
                      className="p-3 bg-zinc-50 dark:bg-zinc-950 rounded-md border border-zinc-200/80 dark:border-zinc-800/80 flex items-center justify-between"
                    >
                      <div className="truncate pr-2">
                        <p className="text-xs font-semibold text-zinc-900 dark:text-zinc-200 truncate">{video.video_name}</p>
                        <p className="text-[11px] text-zinc-400 mt-0.5">
                          {new Date(video.saved_at?.$date || video.saved_at).toLocaleDateString()}
                        </p>
                      </div>
                      <span className="h-1.5 w-1.5 rounded-full bg-teal-500 shrink-0"></span>
                    </div>
                  ))
                ) : (
                  <div className="py-12 text-center border border-dashed border-zinc-200 dark:border-zinc-800 rounded-md">
                    <p className="text-xs text-zinc-400">No projects generated yet.</p>
                  </div>
                )}
              </div>
            </div>

            <Link 
              to="/profile" 
              className="mt-6 w-full text-center py-2 px-4 border border-zinc-200 dark:border-zinc-800 rounded-md text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors"
            >
              View All Projects
            </Link>
          </div>

        </div>
      </div>
    </div>
  );
};

// Login 
const Login = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    if (e) e.preventDefault();
    setError("");

    if (!email || !password) {
      setError("Please enter both email and password.");
      return;
    }

    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailPattern.test(email)) {
      setError("Please enter a valid email address.");
      return;
    }

    setLoading(true);

    try {
      const res = await axios.post(
        `${API_BASE_URL}/auth/login`,
        { email, password },
        { headers: { "Content-Type": "application/json" } }
      );

      localStorage.setItem("token", res.data.access_token);

      const userData = res.data.user;
      if (userData?.profile_photo) {
        userData.avatar = `${API_BASE_URL}/${userData.profile_photo}`;
      }

      localStorage.setItem("userData", JSON.stringify(userData));
      navigate("/profile");

    } catch (err) {
      console.error("Login error:", err.response?.data || err);
      if (err.response?.status === 404) {
        setError("User with this email does not exist.");
      } else if (err.response?.status === 401) {
        setError("Incorrect password. Please try again.");
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError("Login failed. Please try again later.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = () => {
    window.location.href = `${API_BASE_URL}/auth/google/login`;
  };

  const handleGithubLogin = () => {
    window.location.href = `${API_BASE_URL}/auth/github/login`;
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-900 px-4 sm:px-6 pt-20 pb-12 font-sans text-slate-800 dark:text-slate-100">
      <div className="max-w-md w-full bg-white dark:bg-slate-800 p-8 sm:p-10 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700">
        
        {/* Header Section */}
        <div className="text-center mb-8">
          <span className="text-xs font-bold tracking-wider text-indigo-600 dark:text-indigo-400 uppercase bg-indigo-50 dark:bg-indigo-950/50 px-3 py-1.5 rounded-full border border-indigo-200 dark:border-indigo-800/50">
            Account Access
          </span>
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mt-3 mb-1">
            Welcome back
          </h2>
          <p className="text-slate-500 dark:text-slate-400 text-sm">
            Please enter your details to sign in.
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-3.5 bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/60 rounded-xl flex items-center gap-3">
            <svg className="w-5 h-5 text-rose-600 dark:text-rose-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-xs font-medium text-rose-700 dark:text-rose-300">{error}</p>
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-4">
          {/* Email Input */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-300 mb-1">
              Email Address
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 12a4 4 0 10-8 0 4 4 0 008 0zm0 0v1.5a2.5 2.5 0 005 0V12a9 9 0 10-9 9m4.5-1.206a8.959 8.959 0 01-4.5 1.207" />
                </svg>
              </div>
              <input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
                required
              />
            </div>
          </div>

          {/* Password Input */}
          <div>
            <label className="block text-xs font-semibold text-slate-600 dark:text-slate-300 mb-1">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <input
                type={showPassword ? "text" : "password"}
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-10 py-2.5 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
              >
                {showPassword ? (
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.29 3.29m0 0a10.05 10.05 0 015.188-1.583 8.32 8.32 0 013.262.697m0 0l3.29 3.29m-3.29-3.29a10.05 10.05 0 012.308 2.308m0 0c1.275 4.057-2.507 7-7 7a10.05 10.05 0 01-2.308-.273m0 0l-3.29 3.29" />
                  </svg>
                ) : (
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                )}
              </button>
            </div>
          </div>

          {/* Remember Me & Forgot Password */}
          <div className="flex items-center justify-between text-xs pt-1">
            <label className="flex items-center gap-2 cursor-pointer text-slate-600 dark:text-slate-400">
              <input
                type="checkbox"
                className="w-3.5 h-3.5 rounded border-slate-300 dark:border-slate-700 text-indigo-600 focus:ring-indigo-500 bg-slate-50 dark:bg-slate-900"
              />
              <span>Remember me</span>
            </label>
            <button
              type="button"
              onClick={() => navigate("/change-password")}
              className="font-semibold text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              Forgot password?
            </button>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className={`w-full mt-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2.5 rounded-xl transition-colors shadow-sm text-sm flex justify-center items-center gap-2 ${
              loading ? "opacity-70 cursor-wait" : ""
            }`}
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Signing in...</span>
              </>
            ) : (
              "Sign In"
            )}
          </button>
        </form>

        {/* Divider */}
        <div className="my-6 relative flex items-center">
          <div className="flex-grow border-t border-slate-200 dark:border-slate-700"></div>
          <span className="flex-shrink-0 mx-3 text-slate-400 text-xs uppercase font-medium">
            Or continue with
          </span>
          <div className="flex-grow border-t border-slate-200 dark:border-slate-700"></div>
        </div>

        {/* Social Login */}
        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={handleGoogleLogin}
            className="flex items-center justify-center py-2 px-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-semibold text-slate-700 dark:text-slate-200 transition-colors gap-2"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Google
          </button>

          <button
            type="button"
            onClick={handleGithubLogin}
            className="flex items-center justify-center py-2 px-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 hover:bg-slate-100 dark:hover:bg-slate-800 text-xs font-semibold text-slate-700 dark:text-slate-200 transition-colors gap-2"
          >
            <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.379.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.161 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
            </svg>
            GitHub
          </button>
        </div>

        {/* Footer Link */}
        <p className="text-center text-xs text-slate-500 dark:text-slate-400 mt-6">
          Don't have an account?{" "}
          <Link to="/signup" className="text-indigo-600 dark:text-indigo-400 font-semibold hover:underline">
            Create an account
          </Link>
        </p>

      </div>
    </div>
  );
};

// Change Password
const ChangePassword = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleChangePassword = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (!email || !newPassword || !confirmPassword) {
      setError("All fields are required.");
      return;
    }

    if (newPassword !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (newPassword.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }

    setLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/auth/change-password`, {
        email: email,
        new_password: newPassword,
      });

      setSuccess("Password changed successfully! Redirecting to login...");
      setTimeout(() => navigate("/login"), 2500);

    } catch (err) {
      setError(
        err.response?.data?.detail || "Failed to change password. Please check your email."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#E3FDFD] dark:bg-[#222831] px-4 sm:px-6 relative overflow-hidden font-sans pt-20">
      
      {/* Background Glowing Blobs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#00ADB5] opacity-20 dark:opacity-10 blur-[100px] rounded-full pointer-events-none"></div>
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[#71C9CE] opacity-20 dark:opacity-10 blur-[100px] rounded-full pointer-events-none"></div>

      <div className="max-w-md w-full bg-[#CBF1F5]/80 dark:bg-[#393E46]/90 backdrop-blur-xl p-8 sm:p-10 rounded-3xl shadow-2xl border border-[#A6E3E9] dark:border-[#222831] relative z-10 my-8">
        
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-[#E3FDFD] dark:bg-[#222831] border border-[#A6E3E9] dark:border-[#393E46] mb-4 shadow-inner">
            <svg className="w-8 h-8 text-[#00ADB5]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"></path>
            </svg>
          </div>
          <h2 className="text-3xl font-extrabold text-[#222831] dark:text-[#EEEEEE] mb-2 tracking-tight">
            Reset Password
          </h2>
          <p className="text-[#393E46] dark:text-[#A6E3E9] text-sm font-medium">
            Enter your email and create a new secure password.
          </p>
        </div>

        <form onSubmit={handleChangePassword} className="space-y-5">
          
          {/* Email Input */}
          <div>
            <label className="block text-xs font-semibold text-[#393E46] dark:text-[#A6E3E9] mb-1.5 uppercase tracking-wider">
              Email Address
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-[#00ADB5]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 12a4 4 0 10-8 0 4 4 0 008 0zm0 0v1.5a2.5 2.5 0 005 0V12a9 9 0 10-9 9m4.5-1.206a8.959 8.959 0 01-4.5 1.207" />
                </svg>
              </div>
              <input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-11 pr-4 py-3.5 rounded-xl border border-[#A6E3E9] dark:border-[#222831] bg-[#E3FDFD] dark:bg-[#222831] text-[#222831] dark:text-[#EEEEEE] focus:outline-none focus:ring-2 focus:ring-[#00ADB5] transition-all"
                required
              />
            </div>
          </div>

          {/* New Password Input */}
          <div>
            <label className="block text-xs font-semibold text-[#393E46] dark:text-[#A6E3E9] mb-1.5 uppercase tracking-wider">
              New Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-[#00ADB5]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Create new password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="w-full pl-11 pr-12 py-3.5 rounded-xl border border-[#A6E3E9] dark:border-[#222831] bg-[#E3FDFD] dark:bg-[#222831] text-[#222831] dark:text-[#EEEEEE] focus:outline-none focus:ring-2 focus:ring-[#00ADB5] transition-all"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-[#393E46] dark:text-[#A6E3E9] hover:text-[#00ADB5] transition-colors"
              >
                {showPassword ? (
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.29 3.29m0 0a10.05 10.05 0 015.188-1.583 8.32 8.32 0 013.262.697m0 0l3.29 3.29m-3.29-3.29a10.05 10.05 0 012.308 2.308m0 0c1.275 4.057-2.507 7-7 7a10.05 10.05 0 01-2.308-.273m0 0l-3.29 3.29" /></svg>
                ) : (
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
                )}
              </button>
            </div>
          </div>

          {/* Confirm Password Input */}
          <div>
            <label className="block text-xs font-semibold text-[#393E46] dark:text-[#A6E3E9] mb-1.5 uppercase tracking-wider">
              Confirm Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-[#00ADB5]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Confirm new password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full pl-11 pr-12 py-3.5 rounded-xl border border-[#A6E3E9] dark:border-[#222831] bg-[#E3FDFD] dark:bg-[#222831] text-[#222831] dark:text-[#EEEEEE] focus:outline-none focus:ring-2 focus:ring-[#00ADB5] transition-all"
                required
              />
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-3 flex items-start">
              <svg className="w-5 h-5 text-red-500 mr-2 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
              <p className="text-red-500 font-medium text-sm">{error}</p>
            </div>
          )}

          {/* Success Message */}
          {success && (
            <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-3 flex items-start">
              <svg className="w-5 h-5 text-green-500 mr-2 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
              <p className="text-green-600 dark:text-green-400 font-medium text-sm">{success}</p>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className={`w-full bg-[#00ADB5] text-[#EEEEEE] py-3.5 mt-2 rounded-xl font-bold hover:bg-[#71C9CE] hover:shadow-lg hover:-translate-y-0.5 transition-all flex justify-center items-center ${loading ? "opacity-70 cursor-wait" : ""}`}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                Updating...
              </>
            ) : (
              "Update Password"
            )}
          </button>
        </form>

        <p className="text-center text-sm mt-8 text-[#393E46] dark:text-[#A6E3E9]">
          Remember your password?{" "}
          <Link to="/login" className="text-[#00ADB5] font-bold hover:text-[#71C9CE] transition-colors ml-1">
            Back to Login
          </Link>
        </p>

      </div>
    </div>
  );
};

// Signup Page
const Signup = () => {
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [errors, setErrors] = useState({});
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const emailRegex = /^[\w.-]+@[\w.-]+\.\w+$/;

  const validateEmail = (value) => {
    if (!emailRegex.test(value)) return "Enter a valid email address";
    return "";
  };

  const handleGoogleSignup = () => {
    window.location.href = "http://127.0.0.1:8000/auth/google/login";
  };

  const validatePassword = (value) => {
    if (value.length < 8) return "Minimum 8 characters required";
    if (!/[A-Za-z]/.test(value) || !/\d/.test(value))
      return "Must contain letters & numbers";
    return "";
  };

  const handleEmailChange = (e) => {
    const value = e.target.value;
    setEmail(value);
    setErrors((prev) => ({ ...prev, email: validateEmail(value), server: "" }));
  };

  const handlePasswordChange = (e) => {
    const value = e.target.value;
    setPassword(value);
    setErrors((prev) => ({ ...prev, password: validatePassword(value), server: "" }));
  };

  const handleConfirmPassword = (e) => {
    const value = e.target.value;
    setConfirmPassword(value);
    setErrors((prev) => ({
      ...prev,
      confirmPassword: value !== password ? "Passwords do not match" : "",
      server: "",
    }));
  };

  const handleContinue = async () => {
    const newErrors = {
      email: validateEmail(email),
      password: validatePassword(password),
      confirmPassword:
        confirmPassword !== password ? "Passwords do not match" : "",
    };

    setErrors(newErrors);
    if (Object.values(newErrors).some((e) => e)) return;

    sessionStorage.setItem(
      "signupData",
      JSON.stringify({
        username,
        email,
        password,
      })
    );

    navigate("/signup-step2");
  };

  const isDisabled =
    !username ||
    !email ||
    !password ||
    !confirmPassword ||
    Object.values(errors).some((e) => e);

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-zinc-950 px-4 sm:px-6 font-sans text-slate-900 dark:text-zinc-100 py-12">
      <div className="max-w-md w-full bg-white dark:bg-zinc-900 p-8 sm:p-10 rounded-2xl shadow-sm border border-slate-200 dark:border-zinc-800">
        
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-zinc-400">
              Step 1 of 2
            </span>
            <div className="h-1.5 w-24 bg-slate-100 dark:bg-zinc-800 rounded-full overflow-hidden">
              <div className="h-full bg-slate-900 dark:bg-zinc-100 w-1/2 rounded-full" />
            </div>
          </div>
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-zinc-50 tracking-tight">
            Create an account
          </h1>
          <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">
            Join us to start dubbing your videos.
          </p>
        </div>

        {/* Google Signup */}
        <button
          onClick={handleGoogleSignup}
          className="w-full flex items-center justify-center gap-3 py-2.5 px-4 rounded-lg border border-slate-300 dark:border-zinc-700 bg-white dark:bg-zinc-900 text-sm font-medium text-slate-700 dark:text-zinc-200 hover:bg-slate-50 dark:hover:bg-zinc-800/60 transition-all shadow-sm"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24">
            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
          </svg>
          Continue with Google
        </button>

        {/* Divider */}
        <div className="my-6 flex items-center gap-3">
          <div className="flex-grow h-px bg-slate-200 dark:bg-zinc-800" />
          <span className="text-xs text-slate-400 dark:text-zinc-500 uppercase font-medium">Or</span>
          <div className="flex-grow h-px bg-slate-200 dark:bg-zinc-800" />
        </div>

        {/* Inputs */}
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-zinc-300 mb-1.5">
              Username
            </label>
            <input
              type="text"
              placeholder="johndoe"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-lg border border-slate-300 dark:border-zinc-700 bg-white dark:bg-zinc-950 text-slate-900 dark:text-zinc-100 text-sm focus:outline-none focus:ring-2 focus:ring-slate-900 dark:focus:ring-zinc-100 focus:border-transparent transition-all placeholder:text-slate-400"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-zinc-300 mb-1.5">
              Email address
            </label>
            <input
              type="email"
              placeholder="name@company.com"
              value={email}
              onChange={handleEmailChange}
              className={`w-full px-3.5 py-2.5 rounded-lg border ${
                errors.email ? "border-rose-500 focus:ring-rose-500" : "border-slate-300 dark:border-zinc-700 focus:ring-slate-900 dark:focus:ring-zinc-100"
              } bg-white dark:bg-zinc-950 text-slate-900 dark:text-zinc-100 text-sm focus:outline-none focus:ring-2 focus:border-transparent transition-all placeholder:text-slate-400`}
            />
            {errors.email && <p className="text-rose-500 text-xs mt-1.5 font-medium">{errors.email}</p>}
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-zinc-300 mb-1.5">
              Password
            </label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                autoComplete="new-password"
                placeholder="••••••••"
                value={password}
                onChange={handlePasswordChange}
                className={`w-full pl-3.5 pr-12 py-2.5 rounded-lg border ${
                  errors.password ? "border-rose-500 focus:ring-rose-500" : "border-slate-300 dark:border-zinc-700 focus:ring-slate-900 dark:focus:ring-zinc-100"
                } bg-white dark:bg-zinc-950 text-slate-900 dark:text-zinc-100 text-sm focus:outline-none focus:ring-2 focus:border-transparent transition-all placeholder:text-slate-400`}
              />
              <button
                type="button"
                aria-label={showPassword ? "Hide password" : "Show password"}
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3.5 top-1/2 -translate-y-1/2 text-xs font-medium text-slate-500 dark:text-zinc-400 hover:text-slate-800 dark:hover:text-zinc-200 transition-colors"
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
            {errors.password && <p className="text-rose-500 text-xs mt-1.5 font-medium">{errors.password}</p>}
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-zinc-300 mb-1.5">
              Confirm password
            </label>
            <div className="relative">
              <input
                type={showConfirmPassword ? "text" : "password"}
                placeholder="••••••••"
                value={confirmPassword}
                onChange={handleConfirmPassword}
                className={`w-full pl-3.5 pr-12 py-2.5 rounded-lg border ${
                  errors.confirmPassword ? "border-rose-500 focus:ring-rose-500" : "border-slate-300 dark:border-zinc-700 focus:ring-slate-900 dark:focus:ring-zinc-100"
                } bg-white dark:bg-zinc-950 text-slate-900 dark:text-zinc-100 text-sm focus:outline-none focus:ring-2 focus:border-transparent transition-all placeholder:text-slate-400`}
              />
              <button
                type="button"
                aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="absolute right-3.5 top-1/2 -translate-y-1/2 text-xs font-medium text-slate-500 dark:text-zinc-400 hover:text-slate-800 dark:hover:text-zinc-200 transition-colors"
              >
                {showConfirmPassword ? "Hide" : "Show"}
              </button>
            </div>
            {errors.confirmPassword && (
              <p className="text-rose-500 text-xs mt-1.5 font-medium">{errors.confirmPassword}</p>
            )}
          </div>
        </div>

        {/* Server Errors */}
        {errors.server && (
          <div className="mt-4 p-3 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900">
            <p className="text-rose-600 dark:text-rose-400 text-xs font-medium">{errors.server}</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          onClick={handleContinue}
          disabled={isDisabled || loading}
          className={`w-full py-2.5 mt-6 rounded-lg font-medium text-sm transition-all shadow-sm ${
            isDisabled || loading
              ? "bg-slate-100 dark:bg-zinc-800 text-slate-400 dark:text-zinc-600 cursor-not-allowed border border-slate-200 dark:border-zinc-800"
              : "bg-slate-900 dark:bg-zinc-100 text-white dark:text-zinc-900 hover:bg-slate-800 dark:hover:bg-zinc-200 active:scale-[0.99]"
          }`}
        >
          {loading ? "Processing..." : "Continue"}
        </button>

        {/* Login Link */}
        <p className="text-center text-xs text-slate-500 dark:text-zinc-400 mt-6">
          Already have an account?{" "}
          <Link to="/login" className="text-slate-900 dark:text-zinc-100 font-semibold hover:underline">
            Log in
          </Link>
        </p>
      </div>
    </div>
  );
};

const SignupStep2 = () => {
  const navigate = useNavigate();

  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [gender, setGender] = useState("");
  const [dob, setDob] = useState("");
  const [errors, setErrors] = useState({});
  const [age, setAge] = useState(null);
  const [loading, setLoading] = useState(false);

  const validateDob = (value) => {
    if (!value) return "Date of birth is required";

    const dobDate = new Date(value);
    const today = new Date();

    if (dobDate >= today) return "Date of birth cannot be today or future";

    let calculatedAge = today.getFullYear() - dobDate.getFullYear();
    const m = today.getMonth() - dobDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < dobDate.getDate())) {
      calculatedAge--;
    }

    setAge(calculatedAge);
    return "";
  };

  const handleDobChange = (e) => {
    const value = e.target.value;
    setDob(value);
    setErrors((prev) => ({ ...prev, dob: validateDob(value), server: "" }));
  };

  const handleContinue = async () => {
    const dobError = validateDob(dob);
    setErrors((prev) => ({ ...prev, dob: dobError }));
    if (dobError) return;

    const step1Data = JSON.parse(sessionStorage.getItem("signupData"));

    if (!step1Data) {
      navigate("/signup");
      return;
    }

    const finalData = {
      ...step1Data,
      first_name: firstName,
      last_name: lastName,
      gender,
      dob,
    };

    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(finalData),
      });

      const data = await res.json();

      if (!res.ok) {
        setErrors((prev) => ({
          ...prev,
          server: data.detail || "Registration failed. Please try again.",
        }));
        return;
      }

      // Save token and clean storage
      localStorage.setItem("token", data.access_token);
      sessionStorage.removeItem("signupData");

      navigate("/profile");
    } catch (err) {
      console.error("Registration failed", err);
      setErrors((prev) => ({
        ...prev,
        server: "Network error. Please check your connection.",
      }));
    } finally {
      setLoading(false);
    }
  };

  const isDisabled = !firstName || !lastName || !gender || !dob || errors.dob;

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-zinc-950 px-4 sm:px-6 font-sans text-slate-900 dark:text-zinc-100 py-12">
      <div className="max-w-md w-full bg-white dark:bg-zinc-900 p-8 sm:p-10 rounded-2xl shadow-sm border border-slate-200 dark:border-zinc-800">
        
        {/* Top Back Navigation & Progress */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <button
              onClick={() => navigate("/signup")}
              className="inline-flex items-center text-xs font-medium text-slate-500 dark:text-zinc-400 hover:text-slate-900 dark:hover:text-zinc-100 transition-colors gap-1"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
              </svg>
              Back
            </button>
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-zinc-400">
              Step 2 of 2
            </span>
          </div>
          
          <div className="h-1.5 w-full bg-slate-100 dark:bg-zinc-800 rounded-full overflow-hidden mb-6">
            <div className="h-full bg-slate-900 dark:bg-zinc-100 w-full rounded-full transition-all duration-300" />
          </div>

          <h1 className="text-2xl font-semibold text-slate-900 dark:text-zinc-50 tracking-tight">
            Personal details
          </h1>
          <p className="text-sm text-slate-500 dark:text-zinc-400 mt-1">
            Fill in your basic information to complete registration.
          </p>
        </div>

        {/* Inputs */}
        <div className="space-y-4">
          
          {/* First & Last Name */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-slate-700 dark:text-zinc-300 mb-1.5">
                First name
              </label>
              <input
                type="text"
                placeholder="John"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-lg border border-slate-300 dark:border-zinc-700 bg-white dark:bg-zinc-950 text-slate-900 dark:text-zinc-100 text-sm focus:outline-none focus:ring-2 focus:ring-slate-900 dark:focus:ring-zinc-100 focus:border-transparent transition-all placeholder:text-slate-400"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 dark:text-zinc-300 mb-1.5">
                Last name
              </label>
              <input
                type="text"
                placeholder="Doe"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-lg border border-slate-300 dark:border-zinc-700 bg-white dark:bg-zinc-950 text-slate-900 dark:text-zinc-100 text-sm focus:outline-none focus:ring-2 focus:ring-slate-900 dark:focus:ring-zinc-100 focus:border-transparent transition-all placeholder:text-slate-400"
              />
            </div>
          </div>

          {/* Gender */}
          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-zinc-300 mb-1.5">
              Gender
            </label>
            <div className="relative">
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="w-full px-3.5 py-2.5 rounded-lg border border-slate-300 dark:border-zinc-700 bg-white dark:bg-zinc-950 text-slate-900 dark:text-zinc-100 text-sm focus:outline-none focus:ring-2 focus:ring-slate-900 dark:focus:ring-zinc-100 focus:border-transparent transition-all appearance-none pr-10"
              >
                <option value="" disabled>Select your gender</option>
                <option value="female">Female</option>
                <option value="male">Male</option>
                <option value="other">Other</option>
              </select>
              <div className="absolute inset-y-0 right-3.5 flex items-center pointer-events-none text-slate-400">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </div>
          </div>

          {/* Date of Birth */}
          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-zinc-300 mb-1.5">
              Date of birth
            </label>
            <input
              type="date"
              value={dob}
              onChange={handleDobChange}
              className={`w-full px-3.5 py-2.5 rounded-lg border ${
                errors.dob
                  ? "border-rose-500 focus:ring-rose-500"
                  : "border-slate-300 dark:border-zinc-700 focus:ring-slate-900 dark:focus:ring-zinc-100"
              } bg-white dark:bg-zinc-950 text-slate-900 dark:text-zinc-100 text-sm focus:outline-none focus:ring-2 focus:border-transparent transition-all [color-scheme:light] dark:[color-scheme:dark]`}
            />

            {errors.dob && (
              <p className="text-rose-500 text-xs mt-1.5 font-medium">{errors.dob}</p>
            )}

            {age !== null && !errors.dob && (
              <div className="mt-2 inline-flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/40 px-2.5 py-1 rounded-md border border-emerald-200/60 dark:border-emerald-900/60">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
                Verified age: {age} years old
              </div>
            )}
          </div>
        </div>

        {/* Server Errors */}
        {errors.server && (
          <div className="mt-4 p-3 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900">
            <p className="text-rose-600 dark:text-rose-400 text-xs font-medium">{errors.server}</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          onClick={handleContinue}
          disabled={isDisabled || loading}
          className={`w-full py-2.5 mt-6 rounded-lg font-medium text-sm transition-all shadow-sm flex items-center justify-center ${
            isDisabled || loading
              ? "bg-slate-100 dark:bg-zinc-800 text-slate-400 dark:text-zinc-600 cursor-not-allowed border border-slate-200 dark:border-zinc-800"
              : "bg-slate-900 dark:bg-zinc-100 text-white dark:text-zinc-900 hover:bg-slate-800 dark:hover:bg-zinc-200 active:scale-[0.99]"
          }`}
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4 text-current" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Creating Account...
            </span>
          ) : (
            "Complete Registration"
          )}
        </button>
      </div>
    </div>
  );
};

const BASE_URL = "http://127.0.0.1:8000";
// Profile Page
const Profile = () => {
  const [user, setUser] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  const handleProfilePhotoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      const token = localStorage.getItem("token");

      const res = await axios.post(
        "http://127.0.0.1:8000/auth/profile/photo",
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const updatedUser = {
        ...user,
        profile_photo: res.data.profile_photo,
      };

      setUser(updatedUser);
      sessionStorage.setItem("userData", JSON.stringify(updatedUser));
    } catch (err) {
      console.error("Profile upload error:", err);
      alert("Failed to upload profile photo. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  const calculateRemainingTime = (savedAt) => {
    const savedDate = new Date(savedAt?.$date || savedAt).getTime();
    const expiryDate = savedDate + 24 * 60 * 60 * 1000;
    const now = new Date().getTime();
    const diff = expiryDate - now;

    if (diff <= 0) return "Expiring soon";

    const hours = Math.floor(diff / (1000 * 60 * 60));
    const mins = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));

    return `${hours}h ${mins}m left`;
  };

  useEffect(() => {
    const fetchProfileData = async () => {
      try {
        const token = localStorage.getItem("token");
        if (!token) {
          navigate("/login");
          return;
        }
        const res = await axios.get("http://127.0.0.1:8000/auth/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        setUser(res.data);
        setHistory(res.data.saved_videos || []);
        sessionStorage.setItem("userData", JSON.stringify(res.data));
      } catch (err) {
        console.error("Profile Fetch Error:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchProfileData();
  }, [navigate]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-zinc-950">
        <div className="w-8 h-8 border-2 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  const isPremium = user?.plan === "Premium" || user?.plan === "Enterprise";

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 pt-24 pb-16 px-4 md:px-8 font-sans antialiased transition-colors">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* --- HEADER BLOCK --- */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Profile Details Card */}
          <div className="lg:col-span-2 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 md:p-8 flex flex-col sm:flex-row items-center sm:items-start gap-6 relative shadow-sm">
            
            {/* Avatar Section */}
            <div className="relative group flex-shrink-0">
              <img
                src={
                  user?.profile_photo
                    ? `http://127.0.0.1:8000/${user.profile_photo}`
                    : "https://via.placeholder.com/150"
                }
                className="w-28 h-28 rounded-full object-cover border-2 border-zinc-200 dark:border-zinc-700 bg-zinc-100 dark:bg-zinc-800"
                alt="Profile"
              />

              <label className="absolute inset-0 bg-zinc-900/60 rounded-full flex flex-col items-center justify-center text-white opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer">
                <CameraIcon className="w-6 h-6 mb-1" />
                <span className="text-[10px] font-medium tracking-wide">Change</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleProfilePhotoUpload}
                  className="hidden"
                  disabled={uploading}
                />
              </label>

              {uploading && (
                <div className="absolute inset-0 bg-zinc-900/75 rounded-full flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
              )}
            </div>

            {/* Info Section */}
            <div className="text-center sm:text-left space-y-2 flex-1">
              <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2">
                <h1 className="text-2xl font-bold text-zinc-900 dark:text-white tracking-tight">
                  {user?.first_name} {user?.last_name || ""}
                </h1>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
                  isPremium 
                    ? "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20" 
                    : "bg-teal-500/10 text-teal-600 dark:text-teal-400 border-teal-500/20"
                }`}>
                  {isPremium ? "Pro Member" : "Free Plan"}
                </span>
              </div>

              <p className="text-sm text-zinc-500 dark:text-zinc-400 font-mono">@{user?.username || "user"}</p>

              <div className="pt-4 flex flex-wrap justify-center sm:justify-start gap-6 text-xs text-zinc-500 dark:text-zinc-400 border-t border-zinc-100 dark:border-zinc-800 mt-4">
                <div>
                  <span className="block text-zinc-400 dark:text-zinc-500 text-[11px] uppercase tracking-wider font-semibold">Minutes Used</span>
                  <span className="text-sm font-semibold text-zinc-900 dark:text-white">{user?.total_minutes_used || 0} mins</span>
                </div>
                <div>
                  <span className="block text-zinc-400 dark:text-zinc-500 text-[11px] uppercase tracking-wider font-semibold">Account Status</span>
                  <span className="text-sm font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Active
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Subscription Card */}
          <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 flex flex-col justify-between shadow-sm">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500">Current Plan</span>
                {isPremium && <SparklesIcon className="w-5 h-5 text-amber-500" />}
              </div>
              <h2 className="text-3xl font-extrabold text-zinc-900 dark:text-white mb-2">
                {user?.plan || "Free"}
              </h2>
              <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
                {isPremium 
                  ? "Unlimited render access aur high priority processing active hai." 
                  : "Aap basic speed par video dubbing kar rahe hain. Fast rendering ke liye upgrade karein."}
              </p>
            </div>

            {!isPremium ? (
              <button
                onClick={() => navigate("/pricing")}
                className="mt-6 w-full bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 hover:bg-zinc-800 dark:hover:bg-zinc-100 py-2.5 px-4 rounded-xl text-xs font-semibold tracking-wide transition-colors"
              >
                Upgrade Plan
              </button>
            ) : (
              <div className="mt-6 flex items-center gap-2 text-xs text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-500/10 p-3 rounded-xl border border-emerald-200 dark:border-emerald-500/20">
                <CheckIcon className="w-4 h-4 flex-shrink-0" />
                <span>Subscribed & All Features Unlocked</span>
              </div>
            )}
          </div>
        </div>

        {/* --- PROJECTS SECTION --- */}
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800 pb-3">
            <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300 uppercase tracking-wider">
              Recent Dubbing Projects
            </h2>
            <span className="text-xs text-zinc-500 dark:text-zinc-400 font-mono">
              Total: {history.length}
            </span>
          </div>

          {history.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {history.map((video, index) => (
                <div
                  key={index}
                  className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl overflow-hidden hover:border-zinc-300 dark:hover:border-zinc-700 transition-all flex flex-col justify-between shadow-sm"
                >
                  <div className="relative aspect-video bg-zinc-950 border-b border-zinc-200 dark:border-zinc-800">
                    <video
                      src={`http://127.0.0.1:8000/${video.video_url}`}
                      className="w-full h-full object-cover"
                    />
                    <a
                      href={`http://127.0.0.1:8000/${video.video_url}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="absolute bottom-3 right-3 bg-zinc-900/80 hover:bg-zinc-900 text-white p-2 rounded-lg backdrop-blur-md border border-zinc-700 transition-colors"
                      title="Download Video"
                    >
                      <ArrowDownTrayIcon className="w-4 h-4" />
                    </a>
                  </div>

                  <div className="p-4 space-y-3">
                    <h3 className="font-semibold text-sm text-zinc-900 dark:text-zinc-100 truncate">
                      {video.video_name || "Untitled Project"}
                    </h3>

                    {!isPremium && (
                      <div className="flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-400/10 px-2.5 py-1 rounded-md border border-amber-200 dark:border-amber-400/20 w-max">
                        <ClockIcon className="w-3.5 h-3.5" />
                        <span>{calculateRemainingTime(video.saved_at)}</span>
                      </div>
                    )}

                    <div className="flex items-center justify-between text-[11px] text-zinc-400 dark:text-zinc-500 pt-2 border-t border-zinc-100 dark:border-zinc-800">
                      <span>Saved</span>
                      <span>
                        {new Date(video.saved_at?.$date || video.saved_at).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric"
                        })}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl py-16 text-center space-y-4 shadow-sm">
              <VideoCameraIcon className="w-10 h-10 text-zinc-300 dark:text-zinc-600 mx-auto" />
              <div className="space-y-1">
                <h3 className="text-base font-semibold text-zinc-800 dark:text-zinc-200">No projects saved yet</h3>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">Aapka process kiya hua video yahan show hoga.</p>
              </div>
              <Link
                to="/upload"
                className="inline-block bg-teal-500 hover:bg-teal-600 text-white px-5 py-2.5 rounded-xl text-xs font-semibold transition-colors"
              >
                Start New Project
              </Link>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

const Upload = () => {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [user, setUser] = useState(null);
  const [error, setError] = useState("");
  const [targetLanguage, setTargetLanguage] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const data = sessionStorage.getItem("userData");
    if (data) setUser(JSON.parse(data));
  }, []);

  const isPremium = user?.plan === "Premium" || user?.plan === "Enterprise";

  const handleStartDubbing = async () => {
    if (!selectedFile || !targetLanguage) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("language", targetLanguage);

    try {
      setError("");

      // Processing page
      const response = await axios.post(
        "http://127.0.0.1:8000/video/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      console.log("Upload Response:", response.data);

      const jobId = response.data.job_id;

      sessionStorage.setItem("jobId", jobId);

      if (selectedFile) {
        const previewUrl = URL.createObjectURL(selectedFile);

        sessionStorage.setItem("originalVideo", previewUrl);
        sessionStorage.setItem("processedVideo", previewUrl);
      }

      // Go to processing page AFTER jobId is saved
      navigate("/processing");

    } catch (err) {
      console.error(err);

      setError(
        err.response?.data?.detail ||
        "Video processing failed."
      );
    }
  };

  const handleFileChange = (e) => {
    setError("");
    const file = e.target.files[0];
    if (!file) return;

    const validFormats = ["video/mp4", "video/quicktime"];
    if (!validFormats.includes(file.type)) {
      setError("Invalid format. Please upload an MP4 or MOV video file.");
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    const videoElement = document.createElement("video");
    videoElement.preload = "metadata";

    videoElement.onloadedmetadata = () => {
      window.URL.revokeObjectURL(videoElement.src);
      const duration = videoElement.duration;

      if (duration > 30) {
        setError(
          `Video too long. Max allowed duration is 30 seconds. (Current: ${Math.round(
            duration
          )}s)`
        );
        if (fileInputRef.current) fileInputRef.current.value = "";
        return;
      }

      const fileSizeMB = file.size / (1024 * 1024);
      if (!isPremium && fileSizeMB > 100) {
        setError(
          "Free plan limit exceeded (Max 100MB). Upgrade to Premium for larger files."
        );
        if (fileInputRef.current) fileInputRef.current.value = "";
        return;
      }

      setError("");
      setSelectedFile(file);
    };

    videoElement.src = URL.createObjectURL(file);
  };

  const handleCardClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 pt-28 pb-16 px-4 md:px-8 font-sans antialiased transition-colors">
      <div className="max-w-3xl mx-auto space-y-8">
        {/* --- HEADER SECTION --- */}
        <div className="text-center space-y-3">
          {isPremium ? (
            <div className="inline-flex items-center gap-1.5 bg-amber-500/10 text-amber-600 dark:text-amber-400 px-3 py-1 rounded-full border border-amber-500/20 text-xs font-semibold">
              <CheckIcon className="w-3.5 h-3.5" />
              <span>Premium Upload Unlocked</span>
            </div>
          ) : (
            <div className="inline-flex items-center gap-1.5 bg-zinc-200 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 px-3 py-1 rounded-full text-xs font-medium">
              <span>Free Plan • Max 100MB • Up to 30 Seconds</span>
            </div>
          )}

          <h1 className="text-3xl sm:text-4xl font-extrabold text-zinc-900 dark:text-white tracking-tight">
            Upload Your Video
          </h1>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 max-w-lg mx-auto leading-relaxed">
            Select an MP4 or MOV file to translate and dub into your preferred
            language with AI voice matching.
          </p>
        </div>

        {/* --- ERROR MESSAGE --- */}
        {error && (
          <div className="flex items-start gap-3 p-4 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/20 rounded-xl text-rose-700 dark:text-rose-400 text-sm">
            <ExclamationTriangleIcon className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <p className="font-medium">{error}</p>
          </div>
        )}

        {/* --- DRAG & DROP CARD --- */}
        <div
          onClick={handleCardClick}
          className={`relative bg-white dark:bg-zinc-900 border-2 border-dashed rounded-2xl p-8 sm:p-12 text-center transition-all cursor-pointer shadow-sm hover:border-teal-500 dark:hover:border-teal-400 ${
            selectedFile
              ? "border-teal-500/50 dark:border-teal-400/50 bg-teal-500/5"
              : "border-zinc-300 dark:border-zinc-800"
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            onChange={handleFileChange}
            accept="video/mp4, video/quicktime"
          />

          <div className="space-y-4">
            <div className="w-14 h-14 rounded-2xl bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center mx-auto text-zinc-600 dark:text-zinc-300">
              <CloudArrowUpIcon className="w-7 h-7" />
            </div>

            <div className="space-y-1">
              <p className="text-base font-semibold text-zinc-900 dark:text-white">
                Click to browse or drag and drop file here
              </p>
              <p className="text-xs text-zinc-400 dark:text-zinc-500">
                MP4 or MOV formats supported
              </p>
            </div>
          </div>

          {/* Selected File Details */}
          {selectedFile && !error && (
            <div className="mt-6 pt-6 border-t border-zinc-100 dark:border-zinc-800 flex items-center justify-between text-left bg-zinc-50 dark:bg-zinc-800/50 p-4 rounded-xl">
              <div className="flex items-center gap-3 overflow-hidden">
                <VideoCameraIcon className="w-5 h-5 text-teal-500 flex-shrink-0" />
                <div className="truncate">
                  <p className="text-xs font-semibold text-zinc-900 dark:text-white truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-[11px] text-zinc-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <span className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold px-2 py-0.5 rounded bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-200 dark:border-emerald-500/20">
                Ready
              </span>
            </div>
          )}
        </div>

        {/* --- LANGUAGE SELECTION --- */}
        {selectedFile && !error && (
          <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 space-y-3 shadow-sm">
            <label className="block text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              Target Language
            </label>
            <select
              value={targetLanguage}
              onChange={(e) => {
                setTargetLanguage(e.target.value);
                setError("");
              }}
              className="w-full p-3 rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 font-medium transition-all cursor-pointer"
            >
              <option value="">Choose dubbing language...</option>
              <option value="Arabic">Arabic 🇸🇦</option>
              <option value="Spanish">Spanish 🇪🇸</option>
            </select>
          </div>
        )}

        {/* --- ACTION BUTTON --- */}
        <div className="space-y-4 text-center">
          <button
            onClick={handleStartDubbing}
            disabled={!selectedFile || !targetLanguage || error}
            className={`w-full inline-flex items-center justify-center gap-2 font-semibold text-sm py-3.5 px-6 rounded-xl transition-all shadow-sm ${
              !selectedFile || !targetLanguage || error
                ? "bg-zinc-200 dark:bg-zinc-800 text-zinc-400 dark:text-zinc-500 cursor-not-allowed"
                : "bg-teal-500 hover:bg-teal-600 text-white"
            }`}
          >
            <span>
              {isPremium
                ? "Start High-Priority Dubbing"
                : "Start Dubbing Video"}
            </span>
          </button>

          {!isPremium && (
            <button
              onClick={() => navigate("/pricing")}
              className="text-xs text-zinc-500 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
            >
              Need longer video duration?{" "}
              <span className="underline font-medium">Upgrade to Pro</span>
            </button>
          )}
        </div>

        {/* Terms Disclaimer */}
        <p className="text-center text-[11px] text-zinc-400 dark:text-zinc-500 leading-relaxed">
          By uploading, you agree to our Terms of Service.{" "}
          {isPremium
            ? "Videos are stored permanently in your account."
            : "Videos automatically expire after 24 hours on free accounts."}
        </p>
      </div>
    </div>
  );
};

// Processing
const Processing = () => {
  const navigate = useNavigate();

  const [status, setStatus] = useState({
    step: "Uploading Video",
    progress: 0,
    status: "processing",
  });

  const steps = [
    "Uploading Video",
    "Preprocessing Video",
    "Starting AI Pipeline",
    "Extracting Audio",
    "Running Whisper ASR",
    "Translating Text",
    "Generating AI Voice",
    "Preparing Face Detection",
    "Lip Synchronization",
    "Rendering Final Video",
    "Completed",
  ];

  useEffect(() => {
    const jobId = sessionStorage.getItem("jobId");

    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const res = await axios.get(
          `http://127.0.0.1:8000/video/status/${jobId}`
        );

        // Problem 4: Print backend response
        console.log("Backend Response:", res.data);

        setStatus(res.data);

        if (res.data.status === "completed") {
          clearInterval(interval);

          sessionStorage.setItem(
            "processedVideo",
            res.data.video_url
          );

          navigate("/result");
        }

        if (res.data.status === "failed") {
          clearInterval(interval);

          alert(res.data.error || "Processing failed.");

          navigate("/upload");
        }
      } catch (err) {
        console.log("Polling Error:", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [navigate]);

  // Problem 2: Normalized comparison to handle case/spacing mismatches
  const currentIndex = steps.findIndex(
    (s) =>
      s.trim().toLowerCase() ===
      (status.step || "").trim().toLowerCase()
  );

  // Problem 1: Print current step matching details to console
  console.log(
    "Backend Step:",
    status.step,
    "Index:",
    currentIndex
  );

  // Problem 3: Fallback to step 0 if backend step is not found
  const activeIndex = currentIndex === -1 ? 0 : currentIndex;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 flex items-center justify-center px-4 py-20">
      <div className="w-full max-w-4xl bg-white dark:bg-zinc-900 rounded-2xl border border-zinc-200 dark:border-zinc-800 p-8 shadow-sm">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 bg-teal-50 dark:bg-teal-500/10 text-teal-600 dark:text-teal-400 px-4 py-2 rounded-full text-sm font-semibold">
            <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
            AI Pipeline Running
          </div>
          <h1 className="text-3xl font-bold mt-5">
            Processing Your Video
          </h1>
          <p className="text-zinc-500 mt-3">
            Please wait while our AI completes every stage of the dubbing pipeline.
          </p>
        </div>

        {/* Spinner */}
        <div className="flex justify-center mb-6">
          <div className="relative w-24 h-24">
            <div className="absolute inset-0 rounded-full border-4 border-zinc-200 dark:border-zinc-800"></div>
            <div className="absolute inset-0 rounded-full border-4 border-teal-500 border-t-transparent animate-spin"></div>
          </div>
        </div>

        {/* Real-time Progress Bar */}
        <div className="text-center mb-8 max-w-lg mx-auto">
          <p className="font-semibold text-teal-600 dark:text-teal-400 text-lg">
            {status.step}
          </p>

          <p className="text-sm font-medium text-zinc-500 mt-1">
            {status.progress}% Completed
          </p>

          <div className="w-full bg-zinc-200 dark:bg-zinc-800 rounded-full h-2.5 mt-4 overflow-hidden">
            <div
              className="bg-teal-500 h-2.5 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${status.progress}%` }}
            />
          </div>
        </div>

        {/* Steps List */}
        <div className="space-y-3">
          {steps.map((step, index) => {
            // Problem 3 Fix: Using activeIndex instead of currentIndex
            const completed = index < activeIndex;
            const active = index === activeIndex;

            return (
              <div
                key={index}
                className={`flex items-center justify-between rounded-xl border px-5 py-3.5 transition-all ${
                  completed
                    ? "border-emerald-300 bg-emerald-50 dark:bg-emerald-500/10"
                    : active
                    ? "border-teal-400 bg-teal-50 dark:bg-teal-500/10"
                    : "border-zinc-200 dark:border-zinc-800 opacity-60"
                }`}
              >
                <div className="flex items-center gap-4">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all ${
                      completed
                        ? "bg-emerald-500 text-white"
                        : active
                        ? "bg-teal-500 text-white"
                        : "bg-zinc-200 dark:bg-zinc-800 text-zinc-500"
                    }`}
                  >
                    {completed ? (
                      <CheckIcon className="w-4 h-4 stroke-[3]" />
                    ) : (
                      index + 1
                    )}
                  </div>

                  <span
                    className={`font-medium ${
                      active
                        ? "text-teal-600 dark:text-teal-400 font-semibold"
                        : completed
                        ? "text-emerald-700 dark:text-emerald-300"
                        : "text-zinc-600 dark:text-zinc-400"
                    }`}
                  >
                    {step}
                  </span>
                </div>

                {active && (
                  <div className="text-xs font-semibold text-teal-500 animate-pulse bg-teal-500/10 px-3 py-1 rounded-full border border-teal-500/20">
                    Running...
                  </div>
                )}
                {completed && (
                  <div className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                    Done
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <p className="text-center text-xs text-zinc-400 mt-8">
          This page will automatically redirect once the AI pipeline finishes.
        </p>
      </div>
    </div>
  );
};

// Result
const Result = () => {
  const [videoUrl, setVideoUrl] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const originalRef = useRef(null);
  const dubbedRef = useRef(null);

  useEffect(() => {
    const video = sessionStorage.getItem("processedVideo");

    if (video) {
      setVideoUrl(video);
      return;
    }

    const jobId = sessionStorage.getItem("jobId");

    if (!jobId) {
      setErrorMsg("Processed video not found.");
      return;
    }

    const checkVideo = async () => {
      try {
        const res = await axios.get(
          `http://127.0.0.1:8000/video/status/${jobId}`
        );

        if (res.data.status === "completed") {
          sessionStorage.setItem("processedVideo", res.data.video_url);
          setVideoUrl(res.data.video_url);
        } else if (res.data.status === "failed") {
          setErrorMsg(res.data.error || "Processing failed.");
        } else {
          setErrorMsg("Video is still processing.");
        }
      } catch {
        setErrorMsg("Unable to load processed video.");
      }
    };

    checkVideo();
  }, []);

  const handleSyncPlay = (e) => {
    const isOriginal = e.target === originalRef.current;
    const master = isOriginal ? originalRef.current : dubbedRef.current;
    const slave = isOriginal ? dubbedRef.current : originalRef.current;

    if (slave) {
      slave.currentTime = master.currentTime;
      if (master.paused) slave.pause();
      else slave.play().catch(() => {});
    }
  };

  const handleSaveToProfile = async () => {
    const token = localStorage.getItem("token");

    if (!token) {
      setErrorMsg("Please log in to save your video.");
      return;
    }

    setIsSaving(true);
    setErrorMsg("");

    try {
      const response = await fetch(videoUrl);
      const blob = await response.blob();
      const file = new File([blob], "dubbed.mp4", {
        type: "video/mp4",
      });

      const formData = new FormData();
      formData.append("file", file);
      formData.append("video_name", `AI_Dub_${Date.now()}.mp4`);

      await axios.post(
        "http://127.0.0.1:8000/auth/save-video",
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setSaveSuccess(true);
    } catch (err) {
      console.error(err);
      setErrorMsg(
        err.response?.data?.detail || "Unable to save video."
      );
    } finally {
      setIsSaving(false);
    }
  };

  if (!videoUrl) {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-zinc-950 flex flex-col items-center justify-center gap-6 px-4">
        <div className="w-14 h-14 border-4 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-zinc-500 text-sm font-medium text-center">
          {errorMsg || "Loading processed video..."}
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 flex flex-col antialiased pt-16">
      {/* Top Navbar */}
      <header className="h-14 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-semibold text-zinc-900 dark:text-white">
            Dubbing Result
          </h1>
          <span className="text-xs text-zinc-400">
            | Synchronized Preview
          </span>
        </div>

        {/* Main Header Actions */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              const link = document.createElement("a");
              link.href = videoUrl;
              link.download = "Dubbed.mp4";
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
            }}
            className="px-3.5 py-1.5 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:text-zinc-900 dark:hover:text-white bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 rounded-lg transition-colors border border-zinc-200 dark:border-zinc-700"
          >
            Download MP4
          </button>

          <button
            onClick={handleSaveToProfile}
            disabled={isSaving || saveSuccess}
            className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
              saveSuccess
                ? "bg-emerald-50 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/20"
                : "bg-teal-500 hover:bg-teal-600 text-white"
            }`}
          >
            {isSaving
              ? "Saving..."
              : saveSuccess
              ? "Saved to Profile"
              : "Save Video"}
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full flex flex-col gap-6 justify-center">
        {errorMsg && (
          <div className="p-3 text-xs bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/50 text-rose-600 dark:text-rose-300 rounded-lg">
            {errorMsg}
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-6">
          {/* Original Video */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-500 px-1 font-medium">
              <span>Original Source</span>
            </div>
            <div className="bg-black rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 aspect-video shadow-sm">
              <video
                ref={originalRef}
                src={sessionStorage.getItem("originalVideo")}
                onPlay={handleSyncPlay}
                onPause={handleSyncPlay}
                onSeeked={handleSyncPlay}
                controls
                className="w-full h-full object-contain"
              />
            </div>
          </div>

          {/* AI Dubbed Video */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-500 px-1 font-medium">
              <span className="text-teal-600 dark:text-teal-400 font-semibold">
                AI Dubbed
              </span>
            </div>
            <div className="bg-black rounded-xl overflow-hidden border-2 border-teal-500/40 aspect-video shadow-sm">
              <video
                ref={dubbedRef}
                src={videoUrl}
                onPlay={handleSyncPlay}
                onPause={handleSyncPlay}
                onSeeked={handleSyncPlay}
                controls
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        </div>

        <p className="text-center text-xs text-zinc-400 dark:text-zinc-500">
          Playback is synced. Playing or scrubbing either video controls both timelines simultaneously.
        </p>
      </main>
    </div>
  );
};

const PricingCheckIcon = () => (
  <svg className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" strokeWidth="2.5" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
  </svg>
);

// Pricing Page
const Pricing = () => {
  const navigate = useNavigate();
  const [billingCycle, setBillingCycle] = useState("monthly");
  const [user, setUser] = useState(null);

  useEffect(() => {
    const data = sessionStorage.getItem("userData");
    if (data) setUser(JSON.parse(data));
  }, []);

  const plans = [
    {
      id: "Free",
      name: "Free",
      description: "For basic video testing and light usage.",
      price: { monthly: 0, yearly: 0 },
      features: [
        "Up to 5 videos per month",
        "Video saved for 24 hours only",
        "Data auto-deleted after 24h",
        "Standard dubbing speed",
      ],
      buttonText: "Get Started",
    },
    {
      id: "Premium",
      name: "Pro Studio",
      description: "For creators who need full library access and storage.",
      price: { monthly: 19, yearly: 199 },
      features: [
        "Unlimited videos per month",
        "Videos saved permanently in cloud",
        "No database deletion",
        "High-fidelity sync & lip alignment",
        "Priority queue rendering",
      ],
      buttonText: "Upgrade to Pro",
      highlight: true,
    },
    {
      id: "Enterprise",
      name: "Enterprise",
      description: "For teams requiring custom voice models and SLAs.",
      price: { monthly: 499, yearly: 4999 },
      features: [
        "Everything in Pro Studio",
        "Team collaboration tools",
        "Dedicated cloud storage instance",
        "Custom voice cloning options",
        "Strict SLA & 24/7 priority support",
      ],
      buttonText: "Contact Sales",
    },
  ];

  const handleSelectPlan = (planId) => {
    const planObj = plans.find((p) => p.id === planId);
    if (user?.plan === planId) return;

    if (planObj.price.monthly === 0) {
      const userData = JSON.parse(sessionStorage.getItem("userData")) || {};
      userData.plan = "Free";
      sessionStorage.setItem("userData", JSON.stringify(userData));
      navigate("/profile");
      return;
    }

    const amount = billingCycle === "yearly" ? planObj.price.yearly : planObj.price.monthly;
    sessionStorage.setItem(
      "selectedPlan",
      JSON.stringify({ name: planObj.name, duration: billingCycle, amount })
    );
    navigate("/payment");
  };

  return (
    <div className="min-h-screen bg-[#fafafa] dark:bg-[#0a0a0b] text-zinc-900 dark:text-zinc-100 antialiased pt-24 pb-20 px-6">
      <div className="max-w-6xl mx-auto space-y-12">
        
        {/* --- HEADER --- */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-zinc-200 dark:border-zinc-800/80 pb-8">
          <div className="space-y-2 max-w-xl">
            <span className="text-xs font-mono uppercase tracking-widest text-zinc-400">
              Plans & Subscriptions
            </span>
            <h1 className="text-2xl sm:text-3xl font-medium tracking-tight text-zinc-900 dark:text-white">
              Choose your workflow tier
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              {user?.plan 
                ? `Active plan: ${user.plan}. Switch or manage your workspace subscription.` 
                : "Transparent pricing built for solo creators and production teams."}
            </p>
          </div>

          {/* Unified Global Billing Switcher */}
          <div className="inline-flex items-center p-1 bg-zinc-200/60 dark:bg-zinc-900 border border-zinc-300/60 dark:border-zinc-800 rounded-md text-xs font-medium">
            <button
              onClick={() => setBillingCycle("monthly")}
              className={`px-3 py-1.5 rounded transition-all ${
                billingCycle === "monthly" 
                  ? "bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white shadow-sm" 
                  : "text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200"
              }`}
            >
              Monthly billing
            </button>
            <button
              onClick={() => setBillingCycle("yearly")}
              className={`px-3 py-1.5 rounded transition-all flex items-center gap-1.5 ${
                billingCycle === "yearly" 
                  ? "bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white shadow-sm" 
                  : "text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200"
              }`}
            >
              <span>Annual billing</span>
              <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-mono">(-15%)</span>
            </button>
          </div>
        </div>

        {/* --- PRICING GRID --- */}
        <div className="grid md:grid-cols-3 gap-px bg-zinc-200 dark:bg-zinc-800/80 border border-zinc-200 dark:border-zinc-800/80 rounded-lg overflow-hidden shadow-sm">
          {plans.map((plan) => {
            const isCurrentPlan = user?.plan === plan.id || (plan.id === "Free" && !user?.plan);
            const currentPrice = billingCycle === "yearly" ? plan.price.yearly : plan.price.monthly;

            return (
              <div 
                key={plan.id}
                className={`bg-white dark:bg-[#0d0d0e] p-8 flex flex-col justify-between space-y-8 ${
                  plan.highlight ? "relative bg-zinc-50/50 dark:bg-zinc-900/30" : ""
                }`}
              >
                <div className="space-y-6">
                  {/* Title & Status */}
                  <div className="flex items-center justify-between">
                    <h2 className="text-base font-medium text-zinc-900 dark:text-white">
                      {plan.name}
                    </h2>
                    {isCurrentPlan && (
                      <span className="text-[10px] font-mono uppercase bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 px-2 py-0.5 rounded border border-zinc-200 dark:border-zinc-700">
                        Current
                      </span>
                    )}
                  </div>

                  <p className="text-xs text-zinc-500 leading-relaxed min-h-[32px]">
                    {plan.description}
                  </p>

                  {/* Price */}
                  <div className="border-t border-b border-zinc-100 dark:border-zinc-800/60 py-4 flex items-baseline gap-1">
                    <span className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-white">
                      ${currentPrice}
                    </span>
                    <span className="text-xs text-zinc-500">
                      {plan.price.monthly === 0 ? "" : billingCycle === "yearly" ? "/ year" : "/ month"}
                    </span>
                  </div>

                  {/* Features */}
                  {/* Features List */}
                  <ul className="space-y-3 pt-2">
                    {plan.features.map((feature, idx) => (
                      <li key={idx} className="flex items-start gap-3 text-xs text-zinc-600 dark:text-zinc-300">
      
                        {/* Fixed 16px SVG Icon */}
                        <svg 
                          style={{ width: '16px', height: '16px', minWidth: '16px', minHeight: '16px' }}
                          className="text-emerald-500 shrink-0 mt-0.5" 
                          fill="none" 
                          viewBox="0 0 24 24" 
                          strokeWidth="2.5" 
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                        </svg>

                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Button */}
                <button
                  disabled={isCurrentPlan}
                  onClick={() => handleSelectPlan(plan.id)}
                  className={`w-full py-2.5 px-4 rounded text-xs font-medium transition-all ${
                    isCurrentPlan
                      ? "bg-zinc-100 dark:bg-zinc-800/60 text-zinc-400 dark:text-zinc-500 cursor-not-allowed border border-zinc-200 dark:border-zinc-800"
                      : plan.highlight
                      ? "bg-zinc-900 hover:bg-black text-white dark:bg-white dark:hover:bg-zinc-200 dark:text-zinc-900 shadow-sm"
                      : "bg-white hover:bg-zinc-50 text-zinc-900 border border-zinc-300 dark:bg-zinc-900 dark:hover:bg-zinc-800 dark:text-zinc-100 dark:border-zinc-700"
                  }`}
                >
                  {isCurrentPlan ? "Active Subscriptions" : plan.buttonText}
                </button>
              </div>
            );
          })}
        </div>

        {/* --- FOOTER NOTE --- */}
        <p className="text-center text-xs text-zinc-500 pt-4">
          Need custom enterprise limits or bulk API rendering? <a href="#" className="underline underline-offset-4 text-zinc-800 dark:text-zinc-200">Contact our engineering team &rarr;</a>
        </p>

      </div>
    </div>
  );
};

// Payment Page
const PaymentCheckIcon = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
  </svg>
);

const ShieldCheckIcon = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
  </svg>
);

const LockIcon = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
  </svg>
);

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const Payment = () => {
  const navigate = useNavigate();
  const [planData, setPlanData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    try {
      const selectedPlanJSON = sessionStorage.getItem("selectedPlan");

      if (!selectedPlanJSON) {
        console.warn("No plan found in sessionStorage. Using default test plan.");
        setPlanData({ name: "Pro Plan", amount: 29, duration: "monthly" });
        return;
      }

      const parsedPlan = JSON.parse(selectedPlanJSON);

      if (parsedPlan && parsedPlan.amount > 0) {
        setPlanData(parsedPlan);
      } else {
        navigate("/profile");
      }
    } catch (e) {
      console.error("Failed to parse selected plan from storage", e);
      navigate("/pricing");
    }
  }, [navigate]);

  const handlePayment = async () => {
  try {
    setLoading(true);

    const response = await fetch(
      "http://localhost:3001/create-payment",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    const data = await response.json();

    if (data.success) {
      window.location.href = data.checkout_url;
    } else {
      console.error(data.error);
      alert("Payment initialization failed");
    }

  } catch (error) {
    console.error("Payment Error:", error);
    alert("Unable to connect to payment service");
  } finally {
    setLoading(false);
  }
};

  if (!planData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-100 dark:bg-slate-900">
        <div className="text-center text-slate-800 dark:text-slate-100">
          <div className="w-10 h-10 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-xl font-bold mb-1">Loading Checkout...</p>
          <p className="text-sm text-slate-500">Retrieving plan details.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-100 dark:bg-slate-900 px-4 py-20 font-sans">
      {/* Main Container */}
      <div className="flex flex-col-reverse lg:flex-row bg-white dark:bg-slate-800 rounded-3xl shadow-2xl max-w-4xl w-full overflow-hidden border border-slate-200 dark:border-slate-700">
        
        {/* Left Side: Payment Option & Redirect Action */}
        <div className="lg:w-3/5 p-8 lg:p-12 flex flex-col justify-between">
          <div>
            {/* Header */}
            <div className="mb-8">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-950/50 text-indigo-600 dark:text-indigo-400 text-xs font-semibold mb-3 border border-indigo-200 dark:border-indigo-800">
                <ShieldCheckIcon className="w-4 h-4" />
                <span>Encrypted 256-Bit SSL Checkout</span>
              </div>
              <h2 className="text-3xl font-extrabold text-slate-900 dark:text-white mb-2">
                Checkout
              </h2>
              <p className="text-slate-600 dark:text-slate-300 text-sm">
                Complete your payment securely via Safepay to activate your{" "}
                <span className="font-bold text-indigo-600 dark:text-indigo-400">
                  {planData.name}
                </span>.
              </p>
            </div>

            {/* Payment Method Badge */}
            <div className="p-5 bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded-2xl mb-6">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Supported Payment Gateway
                </span>
                <span className="flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                  <PaymentCheckIcon className="w-4 h-4" /> Official Gateway
                </span>
              </div>

              <div className="flex items-center justify-between p-3 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-black text-xs tracking-tighter">
                    SAFE
                  </div>
                  <div>
                    <p className="text-sm font-bold text-slate-900 dark:text-white">
                      Safepay Checkout
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      Visa, Mastercard & Local Cards
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 px-2 py-1 rounded">
                    VISA
                  </span>
                  <span className="text-xs font-semibold bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 px-2 py-1 rounded">
                    Mastercard
                  </span>
                </div>
              </div>
            </div>

            {/* Information Notice */}
            <div className="flex items-start gap-3 p-4 bg-indigo-50/60 dark:bg-indigo-950/30 rounded-xl text-xs text-indigo-900 dark:text-indigo-200 mb-8 border border-indigo-100 dark:border-indigo-900/50">
              <LockIcon className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
              <p>
                You will be redirected to Safepay's secure payment page to complete your purchase. Your payment details are protected by bank-level encryption.
              </p>
            </div>
          </div>

          {/* Action Button */}
          <button
            onClick={handlePayment}
            disabled={loading}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-4 px-6 rounded-xl shadow-lg hover:shadow-indigo-500/25 transition-all flex items-center justify-center gap-2 text-base hover:-translate-y-0.5 disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Redirecting to Safepay...</span>
              </>
            ) : (
              <>
                <LockIcon className="w-5 h-5" />
                <span>
                  Pay PKR {planData.amount.toLocaleString("en-PK")} with Safepay
                </span>
              </>
            )}
          </button>
        </div>

        {/* Right Side: Plan Order Summary */}
        <div className="lg:w-2/5 bg-gradient-to-br from-indigo-600 to-indigo-800 text-white p-8 lg:p-12 flex flex-col justify-between">
          <div>
            <h3 className="text-xl font-bold mb-6 text-white/90">Order Summary</h3>
            
            <div className="bg-white/10 p-6 rounded-2xl backdrop-blur-md border border-white/20 mb-6">
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold text-lg">{planData.name}</span>
                <span className="font-bold text-2xl">${planData.amount}</span>
              </div>
              <p className="text-sm text-indigo-200 capitalize">
                Billing Cycle: {planData.duration}
              </p>
            </div>

            <div className="space-y-4 text-sm text-indigo-100">
              <div className="flex justify-between py-2 border-b border-white/10">
                <span>Subtotal</span>
                <span className="font-semibold">${planData.amount}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-white/10">
                <span>Taxes & Fees</span>
                <span className="font-semibold">$0.00</span>
              </div>
              <div className="flex justify-between text-base pt-2 text-white font-bold">
                <span>Total Amount</span>
                <span className="text-xl">${planData.amount}</span>
              </div>
            </div>
          </div>

          <div className="mt-8 text-xs text-indigo-200/80 space-y-2">
            <p>✔ Instant access to AI pipeline upon payment verification</p>
            <p>✔ Cancel or modify subscription anytime from settings</p>
          </div>
        </div>

      </div>
    </div>
  );
};

// About   
const DEFAULT_AVATAR = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=300&auto=format&fit=crop&q=80";

const About = ({ avatar }) => {
  const memberAvatar = avatar || DEFAULT_AVATAR;

  const features = [
    {
      title: "Fast Processing",
      desc: "Dub a 30-second video in under 90 seconds powered by high-performance GPU infrastructure.",
    },
    {
      title: "Precise Lip-Sync",
      desc: "Achieve accurate frame-level sync while preserving original speaker tones and emotions.",
    },
    {
      title: "Privacy First",
      desc: "Your files are encrypted and automatically deleted after 24 hours to keep your data secure.",
    },
  ];

  const team = [
    { name: "Hunzala Saleem", role: "Co-Founder & CEO", img: memberAvatar },
    { name: "Nida Ilyas", role: "Lead AI Engineer", img: memberAvatar },
    { name: "Ayesha Sarfraz", role: "Creative Director", img: memberAvatar },
    { name: "Laika Sarfraz", role: "Full Stack Developer", img: memberAvatar },
  ];

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 pt-28 pb-20 px-6 font-sans text-slate-800 dark:text-slate-100">
      
      {/* Hero Section */}
      <section className="max-w-3xl mx-auto text-center mb-16">
        <span className="text-xs font-bold tracking-wider text-indigo-600 dark:text-indigo-400 uppercase bg-indigo-50 dark:bg-indigo-950/50 px-3 py-1.5 rounded-full border border-indigo-200 dark:border-indigo-800/50">
          About SyncDub AI
        </span>
        <h1 className="text-4xl sm:text-5xl font-extrabold text-slate-900 dark:text-white mt-4 mb-5 tracking-tight">
          Connecting creators with a global audience
        </h1>
        <p className="text-lg text-slate-600 dark:text-slate-300 leading-relaxed font-normal">
          We help content creators localize their videos effortlessly with natural voice dubbing and accurate lip-sync technology.
        </p>
      </section>

      {/* Mission Section */}
      <section className="max-w-4xl mx-auto mb-20">
        <div className="bg-white dark:bg-slate-800 p-8 sm:p-10 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm text-center">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-3">
            Our Mission
          </h2>
          <p className="text-slate-600 dark:text-slate-300 leading-relaxed max-w-2xl mx-auto text-base">
            Language shouldn't limit your reach. Our goal is to make professional video dubbing fast, accessible, and secure across major global languages including Arabic, Spanish, Hindi, French, and Urdu.
          </p>
        </div>
      </section>

      {/* Features Grid */}
      <section className="max-w-5xl mx-auto grid md:grid-cols-3 gap-6 mb-20">
        {features.map((feat, i) => (
          <div
            key={i}
            className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 hover:border-indigo-500/50 dark:hover:border-indigo-500/50 transition-all shadow-sm"
          >
            <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">
              {feat.title}
            </h3>
            <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed">
              {feat.desc}
            </p>
          </div>
        ))}
      </section>

      {/* Team Section */}
      <section className="max-w-5xl mx-auto mb-20">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white">
            Meet the Team
          </h2>
          <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
            The minds building SyncDub AI
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {team.map((member, i) => (
            <div
              key={i}
              className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 text-center shadow-sm hover:shadow-md transition-shadow"
            >
              <img
                src={member.img}
                alt={member.name}
                className="w-20 h-20 mx-auto rounded-full object-cover mb-4 border border-slate-200 dark:border-slate-700"
                onError={(e) => {
                  e.target.src = DEFAULT_AVATAR;
                }}
              />
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                {member.name}
              </h3>
              <p className="text-xs text-indigo-600 dark:text-indigo-400 font-medium mt-1">
                {member.role}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Call to Action */}
      <section className="max-w-4xl mx-auto bg-indigo-600 dark:bg-indigo-700 rounded-2xl p-8 sm:p-12 text-center text-white shadow-lg">
        <h2 className="text-2xl sm:text-3xl font-bold mb-3">
          Ready to expand your reach?
        </h2>
        <p className="text-indigo-100 text-sm sm:text-base mb-6 max-w-md mx-auto">
          Start dubbing your videos today with AI-powered lip-sync precision.
        </p>
        <Link
          to="/signup"
          className="inline-block bg-white text-indigo-600 font-semibold px-6 py-3 rounded-xl hover:bg-slate-100 transition-colors text-sm shadow-sm"
        >
          Get Started
        </Link>
      </section>

    </div>
  );
};

// Contact Page
const Contact = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });

  const [submitted, setSubmitted] = useState(false);

  const handleChange = (e) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitted(true);
    setTimeout(() => {
      setSubmitted(false);
      setFormData({ name: "", email: "", subject: "", message: "" });
    }, 3000);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 pt-28 pb-20 px-4 sm:px-6 font-sans text-slate-800 dark:text-slate-100">
      <div className="max-w-5xl mx-auto">
        
        {/* Hero Section */}
        <div className="text-center mb-14">
          <span className="text-xs font-bold tracking-wider text-indigo-600 dark:text-indigo-400 uppercase bg-indigo-50 dark:bg-indigo-950/50 px-3 py-1.5 rounded-full border border-indigo-200 dark:border-indigo-800/50">
            Contact Support
          </span>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white mt-4 mb-3 tracking-tight">
            How can we help?
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-base max-w-xl mx-auto">
            Have questions about pricing, API access, or custom dubbing solutions? Send us a message and our team will get back to you shortly.
          </p>
        </div>

        {/* Success Alert */}
        {submitted && (
          <div className="mb-8 p-4 bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-300 rounded-xl text-center text-sm font-medium">
            Thank you! Your message has been sent successfully.
          </div>
        )}

        {/* Main Grid */}
        <div className="grid lg:grid-cols-5 gap-8 items-start">
          
          {/* Form Side (3 cols) */}
          <div className="lg:col-span-3 bg-white dark:bg-slate-800 p-6 sm:p-8 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-6">
              Send a Message
            </h2>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                    Full Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    placeholder="John Doe"
                    className="w-full p-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
                    required
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                    Email Address
                  </label>
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="john@example.com"
                    className="w-full p-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Subject
                </label>
                <input
                  type="text"
                  name="subject"
                  value={formData.subject}
                  onChange={handleChange}
                  placeholder="e.g. Enterprise inquiry"
                  className="w-full p-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Message
                </label>
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  placeholder="How can we assist you?"
                  className="w-full p-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl text-sm text-slate-900 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all h-32 resize-none"
                  required
                ></textarea>
              </div>

              <button
                type="submit"
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-xl transition-colors shadow-sm text-sm flex items-center justify-center gap-2"
              >
                Send Message
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </button>
            </form>
          </div>

          {/* Info Side (2 cols) */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm space-y-6">
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                Contact Details
              </h3>

              <div className="flex items-start gap-4">
                <div className="p-2.5 bg-indigo-50 dark:bg-indigo-950/50 text-indigo-600 dark:text-indigo-400 rounded-lg">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Email</h4>
                  <a href="mailto:support@syncdub.ai" className="text-sm font-medium text-slate-700 dark:text-slate-200 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">
                    support@syncdub.ai
                  </a>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="p-2.5 bg-indigo-50 dark:bg-indigo-950/50 text-indigo-600 dark:text-indigo-400 rounded-lg">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Phone</h4>
                  <a href="tel:+1234567890" className="text-sm font-medium text-slate-700 dark:text-slate-200 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors">
                    +1 (234) 567-8900
                  </a>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="p-2.5 bg-indigo-50 dark:bg-indigo-950/50 text-indigo-600 dark:text-indigo-400 rounded-lg">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Office</h4>
                  <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                    123 Innovation Way, Tech City, CA 90001
                  </p>
                </div>
              </div>
            </div>

            {/* Social Links */}
            <div className="bg-white dark:bg-slate-800 p-6 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm">
              <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-4">
                Follow Us
              </h3>
              <div className="flex gap-3">
                {["X / Twitter", "LinkedIn", "GitHub"].map((social) => (
                  <a
                    key={social}
                    href="#"
                    className="text-xs font-medium px-3 py-2 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-600 dark:text-slate-300 hover:border-indigo-500 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                  >
                    {social}
                  </a>
                ))}
              </div>
            </div>

          </div>

        </div>
      </div>
    </div>
  );
};


function App() {
  useEffect(() => {
  }, []);

  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/change-password" element={<ChangePassword />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/signup-step2" element={<SignupStep2 />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/processing" element={<Processing />} />
        <Route path="/result" element={<Result />} />
        <Route path="/pricing" element={<Pricing />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/payment" element={<Payment />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
