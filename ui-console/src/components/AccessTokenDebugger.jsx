import { useState } from 'react';
import { useAuth } from '../auth/useAuth.js';

const AccessTokenDebugger = () => {
  const { getAccessTokenSilently } = useAuth();
  const [tokenPreview, setTokenPreview] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);

  const requestToken = async () => {
    setStatus('pending');
    setError(null);
    try {
      const token = await getAccessTokenSilently({
        authorizationParams: {
          audience: import.meta.env.VITE_AUTH0_AUDIENCE,
          scope: import.meta.env.VITE_AUTH0_DEFAULT_SCOPE,
        },
      });
      setTokenPreview(`${token.slice(0, 20)}…${token.slice(-20)}`);
      setStatus('success');
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  return (
    <div className="token-debugger">
      <button type="button" className="primary-button" onClick={requestToken} disabled={status === 'pending'}>
        {status === 'pending' ? 'Requesting token…' : 'Request API token'}
      </button>
      {tokenPreview && (
        <p className="token-preview">
          Token preview: <code>{tokenPreview}</code>
        </p>
      )}
      {error && <p className="token-error">Unable to get token: {error}</p>}
    </div>
  );
};

export default AccessTokenDebugger;
