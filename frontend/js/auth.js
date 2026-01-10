// Authentication Logic

// Check if user is logged in
function checkAuth() {
    firebase.auth().onAuthStateChanged((user) => {
        if (!user) {
            // User is not signed in, redirect to login page
            // But only if we are not already on the login page
            if (!window.location.pathname.endsWith('login.html')) {
                window.location.href = 'login.html';
            }
        } else {
            // User is signed in
            console.log("User is signed in:", user.email);
            if (window.location.pathname.endsWith('login.html')) {
                window.location.href = 'index.html';
            }

            // Update UI with user info if element exists
            const userEmailEl = document.getElementById('user-email');
            if (userEmailEl) {
                userEmailEl.textContent = user.email;
            }
        }
    });
}

// Login function
function login(email, password) {
    const errorEl = document.getElementById('login-error');
    const btnForLoading = document.getElementById('login-btn');

    if (btnForLoading) {
        btnForLoading.innerHTML = '<span class="loading-spinner"></span> Signing In...';
        btnForLoading.disabled = true;
    }

    firebase.auth().signInWithEmailAndPassword(email, password)
        .then((userCredential) => {
            // Signed in
            console.log("Login successful");
            window.location.href = 'index.html';
        })
        .catch((error) => {
            console.error("Login Check Error", error);
            if (errorEl) {
                errorEl.style.display = 'block';
                errorEl.textContent = error.message;
            }
            if (btnForLoading) {
                btnForLoading.innerHTML = 'Sign In';
                btnForLoading.disabled = false;
            }
        });
}

// Signup function (optional, for convenience)
function signup(email, password) {
    const errorEl = document.getElementById('login-error');
    const btnForLoading = document.getElementById('signup-btn'); // Assuming a different button or shared

    if (btnForLoading) {
        btnForLoading.innerHTML = '<span class="loading-spinner"></span> Creating Account...';
        btnForLoading.disabled = true;
    }

    firebase.auth().createUserWithEmailAndPassword(email, password)
        .then((userCredential) => {
            // Signed in
            console.log("Signup successful");
            window.location.href = 'index.html';
        })
        .catch((error) => {
            console.error("Signup Error", error);
            if (errorEl) {
                errorEl.style.display = 'block';
                errorEl.textContent = error.message;
            }
            if (btnForLoading) {
                btnForLoading.innerHTML = 'Sign Up'; // Reset text
                btnForLoading.disabled = false;
            }
        });
}

// Logout function
function logout() {
    firebase.auth().signOut().then(() => {
        console.log("Sign-out successful.");
        window.location.href = 'login.html';
    }).catch((error) => {
        console.error("An error happened during sign-out:", error);
    });
}

// Google Login
function loginWithGoogle() {
    const provider = new firebase.auth.GoogleAuthProvider();
    firebase.auth().signInWithPopup(provider)
        .then((result) => {
            window.location.href = 'index.html';
        }).catch((error) => {
            const errorEl = document.getElementById('login-error');
            if (errorEl) {
                errorEl.style.display = 'block';
                errorEl.textContent = error.message;
            }
        });
}
