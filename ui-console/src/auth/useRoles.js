import { useMemo } from 'react';
import { useAuth } from './useAuth.js';

export const useRoles = () => {
  const { user } = useAuth();
  const namespace = import.meta.env.VITE_AUTH0_NAMESPACE ?? 'https://arkcv.com/claims';

  return useMemo(() => {
    if (!user) return [];
    return user[`${namespace}/roles`] ?? [];
  }, [namespace, user]);
};

export default useRoles;
