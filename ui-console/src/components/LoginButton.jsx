import { useAuth } from '../auth/useAuth.js';

const LoginButton = () => {
  const { loginWithRedirect, isLoading } = useAuth();

  return (
    <button
      type="button"
      className="primary-button"
      onClick={() =>
        loginWithRedirect({
          appState: { returnTo: window.location.pathname },
        })
      }
      disabled={isLoading}
    >
      {isLoading ? 'Connecting…' : 'Sign in with Auth0'}
    </button>
  );
};

export default LoginButton;
