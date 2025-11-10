import { useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import ProtectedLayout from '../components/ProtectedLayout.jsx';
import { useAuth } from '../auth/useAuth.js';

const demoQueue = [
  { clipId: 'LC-2307', species: 'Chinook Salmon', confidence: 0.86 },
  { clipId: 'LC-2308', species: 'Steelhead Trout', confidence: 0.62 },
  { clipId: 'LC-2309', species: 'Sockeye Salmon', confidence: 0.64 },
];

const DamPage = () => {
  const { damId } = useParams();
  const { user } = useAuth();
  const namespace = import.meta.env.VITE_AUTH0_NAMESPACE ?? 'https://arkcv.com/claims';
  const assignedDam = user?.[`${namespace}/dam`] ?? 'lower-granite';

  const damLabel = useMemo(
    () =>
      ({
        'lower-granite': 'Lower Granite Dam',
        bonneville: 'Bonneville Dam',
        'chief-joseph': 'Chief Joseph Dam',
      }[damId] ?? damId),
    [damId]
  );

  const isAuthorizedForDam = assignedDam === damId;

  return (
    <ProtectedLayout>
      <section className="card dam-header">
        <div>
          <Link to="/" className="link-button ghost">
            ← Back to dashboard
          </Link>
          <h2>{damLabel}</h2>
          <p>
            Authenticated as <strong>{user?.email ?? user?.name}</strong>
          </p>
          <p className="muted">
            Assigned dam (from ID token): <strong>{assignedDam}</strong>
          </p>
        </div>
      </section>

      {isAuthorizedForDam ? (
        <section className="card">
          <h3>Pending clips (demo data)</h3>
          <table className="queue-table">
            <thead>
              <tr>
                <th>Clip ID</th>
                <th>Species guess</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {demoQueue.map((clip) => (
                <tr key={clip.clipId}>
                  <td>{clip.clipId}</td>
                  <td>{clip.species}</td>
                  <td>{Math.round(clip.confidence * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : (
        <section className="card">
          <h3>Access denied</h3>
          <p>
            Your ID token indicates that you are assigned to <strong>{assignedDam}</strong>. Update the Auth0 app
            metadata or claims if this dam should be visible.
          </p>
        </section>
      )}
    </ProtectedLayout>
  );
};

export default DamPage;
