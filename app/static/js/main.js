/**
 * AutoML Pro - Main JavaScript
 * Shared utilities and common functionality
 */

class AutoMLApp {
    constructor() {
        this.init();
    }

    init() {
        this.initializeParticles();
        this.initializeFlashMessages();
        this.initializeAnimations();
        this.initializeFormValidation();
        this.initializeLoadingStates();
        console.log('🚀 AutoML Pro initialized');
    }

    // ===== PARTICLES SYSTEM =====
    initializeParticles() {
        const particlesContainer = document.querySelector('.particles');
        if (!particlesContainer) return;

        const createParticle = () => {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDuration = (Math.random() * 4 + 4) + 's';
            particle.style.animationDelay = Math.random() * 2 + 's';
            
            particlesContainer.appendChild(particle);

            // Remove particle after animation completes
            setTimeout(() => {
                if (particle.parentNode) {
                    particle.remove();
                }
            }, 10000);
        };

        // Create initial particles
        for (let i = 0; i < 3; i++) {
            setTimeout(createParticle, i * 1000);
        }

        // Continue creating particles
        setInterval(createParticle, 4000);
    }

    // ===== FLASH MESSAGES =====
    initializeFlashMessages() {
        const flashMessages = document.querySelectorAll('.flash');
        if (!flashMessages.length) return;

        // Auto-hide after 5 seconds
        setTimeout(() => {
            flashMessages.forEach(flash => {
                flash.style.opacity = '0';
                flash.style.transform = 'translateY(-10px)';
                setTimeout(() => {
                    if (flash.parentNode) {
                        flash.remove();
                    }
                }, 300);
            });
        }, 5000);

        // Add close button functionality
        flashMessages.forEach(flash => {
            const closeBtn = document.createElement('span');
            closeBtn.innerHTML = '×';
            closeBtn.style.cssText = `
                position: absolute;
                top: 10px;
                right: 15px;
                cursor: pointer;
                font-size: 1.2rem;
                opacity: 0.7;
                transition: opacity 0.3s ease;
            `;
            closeBtn.addEventListener('click', () => {
                flash.style.opacity = '0';
                flash.style.transform = 'translateY(-10px)';
                setTimeout(() => flash.remove(), 300);
            });
            
            flash.style.position = 'relative';
            flash.appendChild(closeBtn);
        });
    }

    // ===== SCROLL ANIMATIONS =====
    initializeAnimations() {
        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    // Add staggered animation for multiple elements
                    const delay = Array.from(entry.target.parentNode.children).indexOf(entry.target) * 100;
                    entry.target.style.animationDelay = delay + 'ms';
                }
            });
        }, observerOptions);

        // Observe all elements with animate-on-scroll class
        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            observer.observe(el);
        });

        // Add hover effects to cards
        document.querySelectorAll('.glass-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-5px) scale(1.02)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0) scale(1)';
            });
        });
    }

    // ===== FORM VALIDATION =====
    initializeFormValidation() {
        const forms = document.querySelectorAll('form[data-validate]');
        
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!this.validateForm(form)) {
                    e.preventDefault();
                    this.showNotification('Veuillez corriger les erreurs dans le formulaire.', 'error');
                }
            });

            // Real-time validation
            const inputs = form.querySelectorAll('.form-control[required]');
            inputs.forEach(input => {
                input.addEventListener('blur', () => {
                    this.validateField(input);
                });
                
                input.addEventListener('input', () => {
                    if (input.classList.contains('error')) {
                        this.validateField(input);
                    }
                });
            });
        });
    }

    validateForm(form) {
        const inputs = form.querySelectorAll('.form-control[required]');
        let isValid = true;

        inputs.forEach(input => {
            if (!this.validateField(input)) {
                isValid = false;
            }
        });

        return isValid;
    }

    validateField(field) {
        const value = field.value.trim();
        let isValid = true;
        let message = '';

        // Required validation
        if (field.hasAttribute('required') && !value) {
            isValid = false;
            message = 'Ce champ est requis.';
        }

        // Email validation
        if (field.type === 'email' && value) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(value)) {
                isValid = false;
                message = 'Veuillez entrer une adresse email valide.';
            }
        }

        // Number validation
        if (field.type === 'number' && value) {
            const min = field.getAttribute('min');
            const max = field.getAttribute('max');
            const numValue = parseFloat(value);
            
            if (min && numValue < parseFloat(min)) {
                isValid = false;
                message = `La valeur doit être supérieure ou égale à ${min}.`;
            }
            
            if (max && numValue > parseFloat(max)) {
                isValid = false;
                message = `La valeur doit être inférieure ou égale à ${max}.`;
            }
        }

        // File validation
        if (field.type === 'file' && field.files.length > 0) {
            const file = field.files[0];
            const allowedTypes = field.getAttribute('accept');
            const maxSize = 16 * 1024 * 1024; // 16MB

            if (file.size > maxSize) {
                isValid = false;
                message = 'Le fichier est trop volumineux (max 16MB).';
            }

            if (allowedTypes) {
                const fileExtension = file.name.split('.').pop().toLowerCase();
                const allowedExtensions = allowedTypes.split(',').map(type => 
                    type.trim().replace('.', '')
                );
                
                if (!allowedExtensions.includes(fileExtension)) {
                    isValid = false;
                    message = `Format non supporté. Utilisez: ${allowedExtensions.join(', ')}`;
                }
            }
        }

        // Update field appearance
        this.updateFieldValidation(field, isValid, message);
        return isValid;
    }

    updateFieldValidation(field, isValid, message) {
        // Remove existing validation
        field.classList.remove('error', 'success');
        const existingMessage = field.parentNode.querySelector('.validation-message');
        if (existingMessage) {
            existingMessage.remove();
        }

        if (!isValid) {
            field.classList.add('error');
            if (message) {
                const messageEl = document.createElement('div');
                messageEl.className = 'validation-message';
                messageEl.textContent = message;
                messageEl.style.cssText = `
                    color: #ff6b6b;
                    font-size: 0.875rem;
                    margin-top: 5px;
                    animation: fadeIn 0.3s ease-out;
                `;
                field.parentNode.appendChild(messageEl);
            }
        } else if (field.value.trim()) {
            field.classList.add('success');
        }
    }

    // ===== LOADING STATES =====
    initializeLoadingStates() {
        // Add loading states to all forms
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', (e) => {
                const submitBtn = form.querySelector('button[type="submit"], input[type="submit"]');
                if (submitBtn && !submitBtn.disabled) {
                    this.showLoadingState(submitBtn);
                }
            });
        });

        // Add loading states to navigation buttons with href
        document.querySelectorAll('a.btn[href]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Don't show loading for external links or anchors
                const href = btn.getAttribute('href');
                if (href && !href.startsWith('#') && !href.startsWith('http')) {
                    this.showLoadingState(btn, 'Chargement...');
                }
            });
        });
    }

    showLoadingState(button, text = 'Traitement...') {
        if (button.dataset.loading === 'true') return;
        
        button.dataset.loading = 'true';
        button.dataset.originalText = button.innerHTML;
        
        button.innerHTML = `
            <span class="spinner"></span>
            ${text}
        `;
        button.disabled = true;
        button.style.transform = 'scale(0.98)';
        
        // Reset after 30 seconds as failsafe
        setTimeout(() => {
            this.resetLoadingState(button);
        }, 30000);
    }

    resetLoadingState(button) {
        if (button.dataset.loading !== 'true') return;
        
        button.innerHTML = button.dataset.originalText || button.innerHTML;
        button.disabled = false;
        button.style.transform = '';
        button.dataset.loading = 'false';
        delete button.dataset.originalText;
    }

    // ===== NOTIFICATIONS =====
    showNotification(message, type = 'info', duration = 4000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            max-width: 400px;
            padding: 15px 20px;
            border-radius: 12px;
            color: white;
            font-weight: 500;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(15px);
            animation: slideInRight 0.3s ease-out;
            cursor: pointer;
        `;

        // Set background based on type
        const backgrounds = {
            success: 'rgba(40, 167, 69, 0.9)',
            error: 'rgba(220, 53, 69, 0.9)',
            warning: 'rgba(255, 193, 7, 0.9)',
            info: 'rgba(23, 162, 184, 0.9)'
        };
        
        notification.style.background = backgrounds[type] || backgrounds.info;
        notification.textContent = message;

        // Add close functionality
        notification.addEventListener('click', () => {
            this.removeNotification(notification);
        });

        document.body.appendChild(notification);

        // Auto-remove
        setTimeout(() => {
            this.removeNotification(notification);
        }, duration);

        // Add CSS for slideInRight animation if not exists
        if (!document.querySelector('#notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                @keyframes slideInRight {
                    from { opacity: 0; transform: translateX(100%); }
                    to { opacity: 1; transform: translateX(0); }
                }
                @keyframes slideOutRight {
                    from { opacity: 1; transform: translateX(0); }
                    to { opacity: 0; transform: translateX(100%); }
                }
                .form-control.error {
                    border-color: #ff6b6b !important;
                    background: rgba(255, 107, 107, 0.1) !important;
                }
                .form-control.success {
                    border-color: #4CAF50 !important;
                    background: rgba(76, 175, 80, 0.1) !important;
                }
            `;
            document.head.appendChild(style);
        }
    }

    removeNotification(notification) {
        if (!notification || !notification.parentNode) return;
        
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }

    // ===== UTILITY METHODS =====
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Octets';
        const k = 1024;
        const sizes = ['Octets', 'Ko', 'Mo', 'Go'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // ===== FILE UPLOAD UTILITIES =====
    initializeFileUpload(dropZone, fileInput, options = {}) {
        const defaultOptions = {
            maxSize: 16 * 1024 * 1024, // 16MB
            allowedTypes: ['csv', 'json', 'xlsx', 'xls'],
            onFileSelect: null,
            onError: null
        };
        
        const config = { ...defaultOptions, ...options };

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelection(files[0], fileInput, config);
            }
        }, false);

        // Handle file input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelection(e.target.files[0], fileInput, config);
            }
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleFileSelection(file, fileInput, config) {
        // Validate file
        if (!this.validateFile(file, config)) {
            return;
        }

        // Update file input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;

        // Trigger custom callback
        if (config.onFileSelect) {
            config.onFileSelect(file);
        }

        this.showNotification(`Fichier sélectionné: ${file.name}`, 'success');
    }

    validateFile(file, config) {
        // Check file size
        if (file.size > config.maxSize) {
            this.showNotification(`Fichier trop volumineux. Taille maximum: ${this.formatFileSize(config.maxSize)}`, 'error');
            return false;
        }

        // Check file type
        const fileExtension = file.name.split('.').pop().toLowerCase();
        if (!config.allowedTypes.includes(fileExtension)) {
            this.showNotification(`Format non supporté. Utilisez: ${config.allowedTypes.join(', ')}`, 'error');
            return false;
        }

        return true;
    }
}

// ===== INITIALIZE APP =====
document.addEventListener('DOMContentLoaded', () => {
    window.automlApp = new AutoMLApp();
});

// ===== EXPORT FOR EXTERNAL USE =====
window.AutoMLUtils = {
    showNotification: (message, type, duration) => window.automlApp?.showNotification(message, type, duration),
    showLoadingState: (button, text) => window.automlApp?.showLoadingState(button, text),
    resetLoadingState: (button) => window.automlApp?.resetLoadingState(button),
    formatFileSize: (bytes) => window.automlApp?.formatFileSize(bytes)
};