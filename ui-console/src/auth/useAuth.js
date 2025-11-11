import { useAuth0 } from '@auth0/auth0-react';

export const useAuth = () => {
  const auth = useAuth0();
  return auth;
};

export default useAuth;
