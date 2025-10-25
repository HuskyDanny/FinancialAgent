import { useState, useEffect } from "react";
import { EnhancedChatInterface } from "./components/EnhancedChatInterface";
import { LoginPage } from "./components/LoginPage";
import HealthPage from "./pages/HealthPage";
import FeedbackPage from "./pages/FeedbackPage";
import { TransactionHistory } from "./pages/TransactionHistory";
import { CreditBalance } from "./components/credits/CreditBalance";
import { authStorage, logout } from "./services/authService";

function App() {
  const [activeTab, setActiveTab] = useState<
    "health" | "chat" | "feedback" | "transactions"
  >("chat");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState("");
  const [isAdmin, setIsAdmin] = useState(false);

  // Check if user is already logged in
  useEffect(() => {
    const token = authStorage.getToken();
    const user = authStorage.getUser();

    if (token && user) {
      setIsAuthenticated(true);
      setUsername(user.username);
      setIsAdmin(user.is_admin || user.username === "allenpan");
    }
  }, []);

  const handleLoginSuccess = () => {
    const user = authStorage.getUser();
    if (user) {
      setUsername(user.username);
      setIsAdmin(user.is_admin || user.username === "allenpan");
    }
    setIsAuthenticated(true);
  };

  const handleLogout = async () => {
    const refreshToken = authStorage.getRefreshToken();

    // Call backend logout to revoke refresh token
    if (refreshToken) {
      try {
        await logout(refreshToken);
      } catch (error) {
        console.error("Logout error:", error);
        // Continue with local logout even if API call fails
      }
    }

    // Clear local storage
    authStorage.clear();
    setIsAuthenticated(false);
    setUsername("");
    setIsAdmin(false);
  };

  // Show login page if not authenticated
  if (!isAuthenticated) {
    return <LoginPage onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Mobile-responsive glassmorphism header */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/70 border-b border-gray-200/50 shadow-sm">
        <div className="mx-auto px-3 sm:px-6 lg:px-8">
          {/* Mobile-first responsive layout */}
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center py-2 gap-2">
            {/* Logo and title */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 sm:gap-3">
                <div className="w-9 h-9 sm:w-11 sm:h-11 bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500 rounded-xl sm:rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/30 ring-2 ring-white/50">
                  <span className="text-xl sm:text-2xl">📊</span>
                </div>
                <div>
                  <h1 className="text-lg sm:text-xl font-bold bg-gradient-to-r from-gray-900 via-blue-900 to-indigo-900 bg-clip-text text-transparent tracking-tight">
                    KlineMatrix
                  </h1>
                  <span className="text-xs font-medium text-gray-500 hidden sm:inline">
                    AI-Powered Financial Intelligence
                  </span>
                </div>
              </div>
              {/* Mobile user info */}
              <div className="flex items-center gap-2 sm:hidden">
                <CreditBalance className="w-32" />
                <button
                  onClick={() => {
                    void handleLogout();
                  }}
                  className="px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-100/80 rounded-lg transition-all"
                >
                  Logout
                </button>
              </div>
            </div>

            {/* Navigation - stacks vertically on mobile */}
            <nav className="flex flex-wrap items-center gap-1.5 sm:gap-2">
              {isAdmin && (
                <button
                  onClick={() => setActiveTab("health")}
                  className={`px-3 sm:px-5 py-1.5 sm:py-2.5 text-xs sm:text-sm font-semibold rounded-lg sm:rounded-xl transition-all duration-200 ${
                    activeTab === "health"
                      ? "bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-lg shadow-blue-500/30"
                      : "text-gray-700 hover:bg-gray-100/80"
                  }`}
                >
                  Health
                </button>
              )}
              <button
                onClick={() => setActiveTab("chat")}
                className={`px-3 sm:px-5 py-1.5 sm:py-2.5 text-xs sm:text-sm font-semibold rounded-lg sm:rounded-xl transition-all duration-200 ${
                  activeTab === "chat"
                    ? "bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-lg shadow-blue-500/30"
                    : "text-gray-700 hover:bg-gray-100/80"
                }`}
              >
                Platform
              </button>
              <button
                onClick={() => setActiveTab("feedback")}
                className={`px-3 sm:px-5 py-1.5 sm:py-2.5 text-xs sm:text-sm font-semibold rounded-lg sm:rounded-xl transition-all duration-200 ${
                  activeTab === "feedback"
                    ? "bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-lg shadow-blue-500/30"
                    : "text-gray-700 hover:bg-gray-100/80"
                }`}
              >
                Feedback
              </button>
              <button
                onClick={() => setActiveTab("transactions")}
                className={`px-3 sm:px-5 py-1.5 sm:py-2.5 text-xs sm:text-sm font-semibold rounded-lg sm:rounded-xl transition-all duration-200 ${
                  activeTab === "transactions"
                    ? "bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-lg shadow-blue-500/30"
                    : "text-gray-700 hover:bg-gray-100/80"
                }`}
              >
                Transactions
              </button>
              {/* Desktop user info */}
              <div className="hidden sm:flex items-center gap-3 pl-4 border-l border-gray-200">
                <CreditBalance className="w-56" />
                <span className="text-sm text-gray-700">👤 {username}</span>
                <button
                  onClick={() => {
                    void handleLogout();
                  }}
                  className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100/80 rounded-lg transition-all"
                >
                  Logout
                </button>
              </div>
            </nav>
          </div>
        </div>
      </header>

      <main className="flex-1 mx-auto w-full">
        {activeTab === "health" && isAdmin && <HealthPage />}

        {activeTab === "chat" && <EnhancedChatInterface />}

        {activeTab === "feedback" && <FeedbackPage />}

        {activeTab === "transactions" && <TransactionHistory />}
      </main>

      <footer className="bg-white border-t">
        <div className="mx-auto py-3 px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            KlineMatrix - AI-Powered Financial Intelligence
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
