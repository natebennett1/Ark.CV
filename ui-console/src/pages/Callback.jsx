import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/useAuth.js';

const CallbackPage = () => {
  const { isLoading, error, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, isLoading, navigate]);

  if (error) {
    return (
      <section className="auth-panel error">
        <h2>Sign-in failed</h2>
        <p>{error.message}</p>
      </section>
    );
  }

  return (
    <section className="auth-panel">
      <h2>Completing sign-in…</h2>
      <p>Finishing Auth0 handshake. You will be redirected automatically.</p>
    </section>
  );
};

export default CallbackPage;
