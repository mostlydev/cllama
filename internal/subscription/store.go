package subscription

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

// DirName is the private directory, inside the auth directory, that holds
// sign-in records. The auth directory itself may be group- or world-writable;
// this one is owner-only and is refused when it is anything else.
const DirName = "subscriptions"

// Record states.
const (
	StateActive         = "active"
	StateSigninRequired = "signin_required"
	StateSignedOut      = "signed_out"
)

// ErrNotSignedIn means no sign-in record exists for the provider.
var ErrNotSignedIn = errors.New("not signed in")

// Record is one plan sign-in. The client ID, host ID and subject outlive a
// sign-out so a later sign-in reuses the same registration.
type Record struct {
	Provider     string    `json:"provider"`
	ClientID     string    `json:"client_id"`
	HostID       string    `json:"host_id"`
	Subject      string    `json:"subject,omitempty"`
	Email        string    `json:"email,omitempty"`
	AccessToken  string    `json:"access_token,omitempty"`
	RefreshToken string    `json:"refresh_token,omitempty"`
	Scopes       []string  `json:"scopes,omitempty"`
	ExpiresAt    time.Time `json:"expires_at,omitempty"`
	SignedInAt   time.Time `json:"signed_in_at,omitempty"`
	RefreshedAt  time.Time `json:"refreshed_at,omitempty"`
	State        string    `json:"state"`
	// Reason says why the record is not active. It never holds token text.
	Reason string `json:"reason,omitempty"`
}

// Usable reports whether the record can authorize a request once refreshed.
func (r *Record) Usable() bool {
	return r != nil && r.State == StateActive && r.ClientID != "" && r.RefreshToken != ""
}

// HasScope reports whether a scope was granted.
func (r *Record) HasScope(scope string) bool {
	for _, s := range r.Scopes {
		if s == scope {
			return true
		}
	}
	return false
}

// Store reads and writes sign-in records under one auth directory.
type Store struct {
	dir string
}

// NewStore returns the store for an auth directory. Nothing is created until
// a record is saved or a lock is taken.
func NewStore(authDir string) *Store {
	return &Store{dir: filepath.Join(authDir, DirName)}
}

// Dir is the private directory holding the records.
func (s *Store) Dir() string { return s.dir }

var providerNameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,63}$`)

func validProvider(provider string) error {
	if !providerNameRe.MatchString(provider) {
		return fmt.Errorf("invalid plan provider name %q", provider)
	}
	return nil
}

// openDir opens the records directory itself, never a link to it, and
// refuses anything but a directory owned by this process's user with
// owner-only permissions. The parent may be writable by others, so every
// operation works through this descriptor: a directory swapped in under the
// same name is either refused here or never seen.
func (s *Store) openDir(create bool) (int, error) {
	open := func() (int, error) {
		return syscall.Open(s.dir, syscall.O_RDONLY|syscall.O_DIRECTORY|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	}
	fd, err := open()
	if errors.Is(err, syscall.ENOENT) {
		if !create {
			return -1, os.ErrNotExist
		}
		if err := os.MkdirAll(filepath.Dir(s.dir), 0o755); err != nil {
			return -1, err
		}
		if err := syscall.Mkdir(s.dir, 0o700); err != nil && !errors.Is(err, syscall.EEXIST) {
			return -1, err
		}
		fd, err = open()
	}
	if err != nil {
		return -1, fmt.Errorf("%s is not a usable directory: %w", s.dir, err)
	}
	var st syscall.Stat_t
	if err := syscall.Fstat(fd, &st); err != nil {
		_ = syscall.Close(fd)
		return -1, err
	}
	if st.Mode&syscall.S_IFMT != syscall.S_IFDIR {
		_ = syscall.Close(fd)
		return -1, fmt.Errorf("%s must be a directory", s.dir)
	}
	if st.Mode&0o777 != 0o700 {
		_ = syscall.Close(fd)
		return -1, fmt.Errorf("%s must have mode 0700, has %04o", s.dir, st.Mode&0o777)
	}
	if int(st.Uid) != os.Geteuid() {
		_ = syscall.Close(fd)
		return -1, fmt.Errorf("%s is owned by uid %d, not by this process (uid %d)", s.dir, st.Uid, os.Geteuid())
	}
	return fd, nil
}

// openOwned opens a name inside the pinned directory without following a
// link, and requires a regular file owned by this user that nobody else can
// read.
func (s *Store) openOwned(dirfd int, name string, flags int) (*os.File, error) {
	// O_NONBLOCK keeps a FIFO planted under a record name from hanging the open.
	var fd int
	var err error
	for attempt := 0; ; attempt++ {
		fd, err = unix.Openat(dirfd, name, flags|unix.O_NOFOLLOW|unix.O_CLOEXEC|unix.O_NONBLOCK, 0o600)
		// Some kernels report a concurrent create of the same name as
		// absent. With O_CREAT that answer is never final.
		if err == nil || flags&syscall.O_CREAT == 0 || flags&syscall.O_EXCL != 0 || !errors.Is(err, syscall.ENOENT) || attempt >= 20 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	if err != nil {
		return nil, err
	}
	var st syscall.Stat_t
	if err := syscall.Fstat(fd, &st); err != nil {
		_ = syscall.Close(fd)
		return nil, err
	}
	if st.Mode&syscall.S_IFMT != syscall.S_IFREG || st.Mode&0o077 != 0 || int(st.Uid) != os.Geteuid() {
		_ = syscall.Close(fd)
		return nil, fmt.Errorf("%s must be a regular file readable only by its owner", name)
	}
	return os.NewFile(uintptr(fd), name), nil
}

// Load returns a provider's record, or ErrNotSignedIn.
func (s *Store) Load(provider string) (*Record, error) {
	if err := validProvider(provider); err != nil {
		return nil, err
	}
	dirfd, err := s.openDir(false)
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotSignedIn
	}
	if err != nil {
		return nil, err
	}
	defer syscall.Close(dirfd)
	f, err := s.openOwned(dirfd, provider+".json", syscall.O_RDONLY)
	if errors.Is(err, syscall.ENOENT) {
		return nil, ErrNotSignedIn
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, 1<<20))
	if err != nil {
		return nil, err
	}
	var rec Record
	if err := json.Unmarshal(data, &rec); err != nil {
		return nil, fmt.Errorf("sign-in record for %s is not valid", provider)
	}
	if rec.Provider != provider {
		return nil, fmt.Errorf("sign-in record for %s names another provider", provider)
	}
	return &rec, nil
}

// Save atomically replaces a provider's record. All fields of a refreshed
// token set are written together, and a write that fails leaves the previous
// record in place.
func (s *Store) Save(rec *Record) error {
	if rec == nil {
		return errors.New("no sign-in record")
	}
	if err := validProvider(rec.Provider); err != nil {
		return err
	}
	dirfd, err := s.openDir(true)
	if err != nil {
		return err
	}
	defer syscall.Close(dirfd)
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	suffix := make([]byte, 6)
	if _, err := rand.Read(suffix); err != nil {
		return err
	}
	dest := rec.Provider + ".json"
	tmp := fmt.Sprintf("%s.%x.tmp", dest, suffix)
	f, err := s.openOwned(dirfd, tmp, syscall.O_WRONLY|syscall.O_CREAT|syscall.O_EXCL)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = unix.Renameat(dirfd, tmp, dirfd, dest)
	}
	if err != nil {
		_ = unix.Unlinkat(dirfd, tmp, 0)
		return err
	}
	return syscall.Fsync(dirfd)
}

// Lock serializes refresh and sign-in commits for one provider across
// processes. A rotating refresh token must never be spent by two holders. The
// wait ends when ctx does.
func (s *Store) Lock(ctx context.Context, provider string) (func(), error) {
	if err := validProvider(provider); err != nil {
		return nil, err
	}
	dirfd, err := s.openDir(true)
	if err != nil {
		return nil, err
	}
	defer syscall.Close(dirfd)
	f, err := s.openOwned(dirfd, provider+".lock", syscall.O_RDWR|syscall.O_CREAT)
	if err != nil {
		return nil, err
	}
	for {
		err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			break
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) && !errors.Is(err, syscall.EINTR) {
			_ = f.Close()
			return nil, err
		}
		select {
		case <-ctx.Done():
			_ = f.Close()
			return nil, ctx.Err()
		case <-time.After(25 * time.Millisecond):
		}
	}
	return func() {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		_ = f.Close()
	}, nil
}
