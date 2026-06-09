import { useState } from 'react'
import './App.css'
import { Sidebar } from './components/Sidebar'
import { Header } from './components/Header'
import { Dashboard } from './screens/Dashboard'
import { Generator } from './screens/Generator'
import { History } from './screens/History'
import { Training } from './screens/Training'
import { Analytics } from './screens/Analytics'
import { Settings } from './screens/Settings'
import { Help } from './screens/Help'
import { AppProvider } from './context/AppContext'

function App() {
  const [activeScreen, setActiveScreen] = useState('dashboard')

  const screenTitles: Record<string, { title: string; subtitle?: string }> = {
    dashboard: { title: 'Dashboard', subtitle: 'Overview of your AI P&ID screen generation activity' },
    generator: { title: 'AI Screen Generator', subtitle: 'Generate Ignition Perspective screens from P&ID drawings' },
    history: { title: 'Generation History', subtitle: 'View and manage your past screen generations' },
    training: { title: 'AI Training', subtitle: 'Train custom component detection models' },
    analytics: { title: 'Analytics', subtitle: 'Insights and statistics about your usage' },
    settings: { title: 'Settings', subtitle: 'Configure your application preferences' },
    help: { title: 'Help', subtitle: 'Get help and support' },
  }

  const renderScreen = () => {
    switch (activeScreen) {
      case 'dashboard':
        return <Dashboard />
      case 'generator':
        return <Generator />
      case 'history':
        return <History />
      case 'training':
        return <Training />
      case 'analytics':
        return <Analytics />
      case 'settings':
        return <Settings />
      case 'help':
        return <Help />
      default:
        return <Dashboard />
    }
  }

  return (
    <AppProvider>
      <div className="app">
        <Sidebar activeScreen={activeScreen} onScreenChange={setActiveScreen} />
        <div className="main-content">
          <Header 
            title={screenTitles[activeScreen]?.title || 'Dashboard'} 
            subtitle={screenTitles[activeScreen]?.subtitle}
          />
          <main className="screen-container">
            {renderScreen()}
          </main>
        </div>
      </div>
    </AppProvider>
  )
}

export default App
