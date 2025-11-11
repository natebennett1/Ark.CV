import PropTypes from 'prop-types'
import { Auth0Provider } from '@auth0/auth0-react'
import { useNavigate } from 'react-router-dom'

const normalizeBaseUrl = (value) => {
  if (!value) return window.location.origin
  return value.endsWith('/') ? value.slice(0, -1) : value
}

const Auth0ProviderWithNavigate = ({ children }) => {
  const domain = import.meta.env.VITE_AUTH0_DOMAIN
  const clientId = import.meta.env.VITE_AUTH0_CLIENT_ID
  const audience = import.meta.env.VITE_AUTH0_AUDIENCE
  const scope =
    import.meta.env.VITE_AUTH0_DEFAULT_SCOPE ?? 'openid profile email read:clips read:fishcounts'
  const baseUrl = normalizeBaseUrl(import.meta.env.VITE_AUTH0_APP_BASE_URL)
  const redirectUri = `${baseUrl}/callback`

  const navigate = useNavigate()

  if (!domain || !clientId) {
    return (
      <div className="auth-panel">
        <h2>Auth0 is not configured</h2>
        <p>
          Set <code>VITE_AUTH0_DOMAIN</code> and <code>VITE_AUTH0_CLIENT_ID</code> in your environment.
        </p>
      </div>
    )
  }

  const onRedirectCallback = (appState) => {
    navigate(appState?.returnTo || '/', { replace: true })
  }

  return (
    <Auth0Provider
      domain={domain}
      clientId={clientId}
      authorizationParams={{
        redirect_uri: redirectUri,
        scope,
        ...(audience ? { audience } : {}),
      }}
      cacheLocation="localstorage"
      useRefreshTokens
      onRedirectCallback={onRedirectCallback}
    >
      {children}
    </Auth0Provider>
  )
}

Auth0ProviderWithNavigate.propTypes = {
  children: PropTypes.node.isRequired,
}

export default Auth0ProviderWithNavigate
