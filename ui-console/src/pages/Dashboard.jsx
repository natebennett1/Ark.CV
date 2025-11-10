import { Link } from 'react-router-dom';
import ProtectedLayout from '../components/ProtectedLayout.jsx';
import { useAuth } from '../auth/useAuth.js';
import { useRoles } from '../auth/useRoles.js';
import AccessTokenDebugger from '../components/AccessTokenDebugger.jsx';

const sampleDams = [
  { id: 'lower-granite', name: 'Lower Granite Dam', total: 12842 },
  { id: 'bonneville', name: 'Bonneville Dam', total: 15403 },
  { id: 'chief-joseph', name: 'Chief Joseph Dam', total: 8421 },
];

const Dashboard = () => {
  const { user } = useAuth();
  const roles = useRoles();

  return (
    <ProtectedLayout>
      <section className="card">
        <h2>Welcome back{user?.given_name ? `, ${user.given_name}` : ''}!</h2>
        <p>
          This React prototype is wired to Auth0. Use it to verify login/logout flows and protect downstream
          dam-specific routes.
        </p>
        <div className="roles-row">
          <span>Your roles:</span>
          {roles.length === 0 ? (
            <span className="role-chip muted">No roles detected</span>
          ) : (
            roles.map((role) => (
              <span key={role} className="role-chip">
                {role}
              </span>
            ))
          )}
        </div>
        <AccessTokenDebugger />
      </section>
      <section className="card">
        <h3>Dam queues</h3>
        <p>Only users assigned to a dam should see these clips once API integration is in place.</p>
        <ul className="dam-list">
          {sampleDams.map((dam) => (
            <li key={dam.id} className="dam-list__item">
              <div>
                <strong>{dam.name}</strong>
                <span>{dam.total.toLocaleString()} fish counted YTD</span>
              </div>
              <Link className="link-button" to={`/dams/${dam.id}`}>
                Open queue →
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </ProtectedLayout>
  );
};

export default Dashboard;
