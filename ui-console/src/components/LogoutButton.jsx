import { useAuth } from '../auth/useAuth.js';

const LogoutButton = () => {
  const { logout } = useAuth();
  const baseUrl = import.meta.env.VITE_AUTH0_APP_BASE_URL ?? window.location.origin;

  return (
    <button
      type="button"
      className="ghost-button"
      onClick={() =>
        logout({
          logoutParams: { returnTo: baseUrl },
        })
      }
    >
      Log out
    </button>
  );
};

export default LogoutButton;
