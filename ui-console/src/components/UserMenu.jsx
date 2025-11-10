import LogoutButton from './LogoutButton.jsx';
import { useAuth } from '../auth/useAuth.js';

const UserMenu = () => {
  const { user } = useAuth();

  if (!user) return null;

  return (
    <div className="user-menu">
      <div>
        <span className="user-name">{user.name ?? user.email}</span>
        {user.email && <span className="user-email">{user.email}</span>}
      </div>
      <LogoutButton />
    </div>
  );
};

export default UserMenu;
