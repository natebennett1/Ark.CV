import PropTypes from 'prop-types';
import UserMenu from './UserMenu.jsx';

const ProtectedLayout = ({ children }) => (
  <div className="app-shell">
    <header className="top-bar">
      <div>
        <h1>Ark.CV HITL Console</h1>
        <p>Authenticated dashboard prototype</p>
      </div>
      <UserMenu />
    </header>
    <main>{children}</main>
  </div>
);

ProtectedLayout.propTypes = {
  children: PropTypes.node.isRequired,
};

export default ProtectedLayout;
