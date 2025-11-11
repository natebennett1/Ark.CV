import { Navigate, Route, Routes } from 'react-router-dom'
import RequireAuth from './auth/RequireAuth.jsx'
import Dashboard from './pages/Dashboard.jsx'
import DamPage from './pages/Dam.jsx'
import LoginPage from './pages/Login.jsx'
import CallbackPage from './pages/Callback.jsx'
import './App.css'

const App = () => (
  <Routes>
    <Route path="/login" element={<LoginPage />} />
    <Route path="/callback" element={<CallbackPage />} />
    <Route
      path="/"
      element={
        <RequireAuth>
          <Dashboard />
        </RequireAuth>
      }
    />
    <Route
      path="/dams/:damId"
      element={
        <RequireAuth>
          <DamPage />
        </RequireAuth>
      }
    />
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes>
)

export default App
