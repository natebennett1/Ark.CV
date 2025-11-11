import PropTypes from 'prop-types';
import LoginButton from '../components/LoginButton.jsx';
import { useAuth } from './useAuth.js';

const RequireAuth = ({ children }) => {
  const { isAuthenticated, isLoading, error } = useAuth();

  if (isLoading) {
    return (
      <section className="auth-panel">
        <h2>Checking your session…</h2>
        <p>Hold tight while we confirm your access.</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="auth-panel error">
        <h2>Authentication error</h2>
        <p>{error.message}</p>
        <LoginButton />
      </section>
    );
  }

  if (!isAuthenticated) {
    return (
      <section className="auth-panel">
        <h2>Sign in required</h2>
        <p>You must be an authorized Ark.CV operator to view this area.</p>
        <LoginButton />
      </section>
    );
  }

  return children;
};

RequireAuth.propTypes = {
  children: PropTypes.node.isRequired,
};

export default RequireAuth;
