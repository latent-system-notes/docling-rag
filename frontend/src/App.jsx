import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'
import ChangePasswordPage from './pages/ChangePasswordPage'
import SearchPage from './pages/SearchPage'
import DocumentsPage from './pages/DocumentsPage'
import UsersPage from './pages/UsersPage'
import GroupsPage from './pages/GroupsPage'
import PermissionsPage from './pages/PermissionsPage'
import SettingsPage from './pages/SettingsPage'
import IngestionPage from './pages/IngestionPage'

function ProtectedRoute({ children }) {
  const token = localStorage.getItem('token')
  if (!token) return <Navigate to="/login" replace />
  const user = JSON.parse(localStorage.getItem('user') || '{}')
  if (user.must_change_password) return <Navigate to="/change-password" replace />
  return children
}

function AdminRoute({ children }) {
  const user = JSON.parse(localStorage.getItem('user') || '{}')
  if (!user.is_admin) return <Navigate to="/search" replace />
  return children
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/change-password" element={<ChangePasswordPage />} />
      <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        {/* Default: everyone goes to search */}
        <Route index element={<Navigate to="/search" replace />} />

        {/* All users */}
        <Route path="search" element={<SearchPage />} />
        <Route path="documents" element={<DocumentsPage />} />

        {/* Admin only */}
        <Route path="users" element={<AdminRoute><UsersPage /></AdminRoute>} />
        <Route path="groups" element={<AdminRoute><GroupsPage /></AdminRoute>} />
        <Route path="permissions" element={<AdminRoute><PermissionsPage /></AdminRoute>} />
        <Route path="settings" element={<AdminRoute><SettingsPage /></AdminRoute>} />
        <Route path="ingestion" element={<AdminRoute><IngestionPage /></AdminRoute>} />
      </Route>
    </Routes>
  )
}
