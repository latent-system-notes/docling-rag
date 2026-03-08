import { useState, useEffect } from 'react'
import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { Users, Shield, FolderTree, LogOut, Search, FileText, Menu, X, PanelLeftClose } from 'lucide-react'

export default function Layout() {
  const navigate = useNavigate()
  const user = JSON.parse(localStorage.getItem('user') || '{}')
  const isAdmin = user.is_admin === true
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const check = () => {
      const mobile = window.innerWidth <= 768
      setIsMobile(mobile)
      if (mobile) setSidebarOpen(false)
      else setSidebarOpen(true)
    }
    check()
    window.addEventListener('resize', check)
    return () => window.removeEventListener('resize', check)
  }, [])

  const logout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/login')
  }

  const handleNavClick = () => {
    if (isMobile) setSidebarOpen(false)
  }

  // Show hamburger only when sidebar is closed
  const showHamburger = !sidebarOpen

  return (
    <div className="layout">
      {showHamburger && (
        <button className="sidebar-toggle visible" onClick={() => setSidebarOpen(true)} title="Open sidebar">
          <Menu size={20} />
        </button>
      )}

      {isMobile && sidebarOpen && <div className="sidebar-overlay visible" onClick={() => setSidebarOpen(false)} />}

      <nav className={`sidebar ${sidebarOpen ? 'open' : 'collapsed'}`}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 1.5rem', marginBottom: '2rem' }}>
          <h2 style={{ margin: 0 }}>Docling RAG</h2>
          <button className="btn-icon" onClick={() => setSidebarOpen(false)} title="Close sidebar">
            {isMobile ? <X size={18} /> : <PanelLeftClose size={18} />}
          </button>
        </div>

        <NavLink to="/search" className={({ isActive }) => isActive ? 'active' : ''} onClick={handleNavClick}>
          <Search size={18} /> Search
        </NavLink>
        <NavLink to="/documents" className={({ isActive }) => isActive ? 'active' : ''} onClick={handleNavClick}>
          <FileText size={18} /> Documents
        </NavLink>

        {isAdmin && (
          <>
            <div style={{ padding: '0.75rem 1.5rem 0.25rem', fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-dim)' }}>
              Administration
            </div>
            <NavLink to="/users" className={({ isActive }) => isActive ? 'active' : ''} onClick={handleNavClick}>
              <Users size={18} /> Users
            </NavLink>
            <NavLink to="/groups" className={({ isActive }) => isActive ? 'active' : ''} onClick={handleNavClick}>
              <Shield size={18} /> Groups
            </NavLink>
            <NavLink to="/permissions" className={({ isActive }) => isActive ? 'active' : ''} onClick={handleNavClick}>
              <FolderTree size={18} /> Permissions
            </NavLink>
          </>
        )}

        <div className="sidebar-footer">
          <div className="text-sm text-muted mb-2">
            {user.username || 'user'}
            {isAdmin && <span className="badge badge-green" style={{ marginLeft: '0.5rem' }}>admin</span>}
          </div>
          <button className="btn-danger btn-sm" onClick={logout} style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
            <LogOut size={14} /> Logout
          </button>
        </div>
      </nav>

      <main className={`main ${!sidebarOpen ? 'sidebar-collapsed' : ''}`}>
        <Outlet />
      </main>
    </div>
  )
}
