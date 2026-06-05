import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Predictions from './pages/Predictions'
import Rankings from './pages/Rankings'
import Explanations from './pages/Explanations'

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Navbar */}
        <nav className="navbar border-b">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <Link to="/" className="text-2xl font-bold text-blue-600">
              ⛳ PGA Analysis
            </Link>
            <div className="flex gap-6">
              <Link to="/" className="hover:text-blue-600">
                Dashboard
              </Link>
              <Link to="/predictions" className="hover:text-blue-600">
                Predictions
              </Link>
              <Link to="/rankings" className="hover:text-blue-600">
                Rankings
              </Link>
              <Link to="/explanations" className="hover:text-blue-600">
                Explanations
              </Link>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/rankings" element={<Rankings />} />
            <Route path="/explanations" element={<Explanations />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-gray-100 border-t mt-12">
          <div className="max-w-7xl mx-auto px-4 py-6 text-center text-gray-600">
            <p>PGA Analysis API v1.0 | ML-powered golf predictions</p>
          </div>
        </footer>
      </div>
    </Router>
  )
}