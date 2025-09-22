/**
 * Portfolio - JavaScript Principal
 * Version moderne sans jQuery
 */

class Portfolio {
    constructor() {
        this.init();
    }

    init() {
        // Initialisation une fois le DOM chargé
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.onDOMReady());
        } else {
            this.onDOMReady();
        }

        // Initialisation après chargement complet (images, etc.)
        window.addEventListener('load', () => this.onWindowLoad());
    }

    onDOMReady() {
        this.loadNavigationPartial();
        this.initNavigation();
        this.initMediaControls();
        this.initResponsiveColumns();
    }

    onWindowLoad() {
        this.hideLoadingScreen();
    }

    // Gestion de l'écran de chargement
    hideLoadingScreen() {
        const loading = document.getElementById('loading');
        if (loading) {
            loading.classList.add('hidden');
            setTimeout(() => loading.remove(), 300);
        }
    }

    // Gestion de la navigation
    loadNavigationPartial() {
        const navContainer = document.querySelector('[data-include="nav"], #nav, #nav-root');
        if (!navContainer) {
            return;
        }

        // Charger le fragment de navigation commun
        fetch('nav.html', { cache: 'no-store' })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load nav.html');
                }
                return response.text();
            })
            .then(html => {
                navContainer.innerHTML = html;
                this.initNavigation();
            })
            .catch(() => {
                // En cas d'échec, ne rien faire: laisser le DOM tel quel
                // et tenter quand même d'initialiser la navigation existante
                this.initNavigation();
            });
    }

    initNavigation() {
        // Marquer la page active
        const currentPage = window.location.pathname.split('/').pop().replace('.html', '');
        const activeLink = document.querySelector(`[data-page="${currentPage}"]`);
        
        if (activeLink) {
            activeLink.classList.add('active');
        }

        // Les liens n'ont pas besoin d'indicateur de chargement supplémentaire
        // L'écran de chargement est déjà dans chaque page HTML
    }

    // Gestion des contrôles média
    initMediaControls() {
        const mediaControls = document.querySelectorAll('.media-controls');
        
        mediaControls.forEach(control => {
            const buttons = control.querySelectorAll('.media-button');
            const media = control.previousElementSibling.querySelector('img, video');
            
            buttons.forEach(button => {
                button.addEventListener('click', (e) => {
                    this.handleMediaButtonClick(e, buttons, media);
                });
            });
        });
    }

    handleMediaButtonClick(event, allButtons, mediaElement) {
        const button = event.target;
        const target = button.getAttribute('data-target');
        
        // Mettre à jour l'état actif des boutons
        allButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        
        // Changer la source du média selon le contexte
        if (mediaElement) {
            const newSource = this.getMediaSource(target, mediaElement);
            if (newSource) {
                if (mediaElement.tagName === 'IMG') {
                    mediaElement.src = newSource;
                } else if (mediaElement.tagName === 'VIDEO') {
                    const source = mediaElement.querySelector('source');
                    if (source) {
                        source.src = newSource;
                        mediaElement.load();
                    }
                }
            }
        }
    }

    getMediaSource(target, mediaElement) {
        // Logique pour déterminer la nouvelle source selon le projet
        // À personnaliser selon les besoins de chaque page
        const basePath = '../medias/';
        
        // Exemples de mappings
        const mediaMappings = {
            // Chaturbate cartography
            'viewing': 'viewing.png',
            'donating': 'donating.png',
            
            // Dynamics of massive microcosms
            'dmn_relative': 'dmn_relative_2000p.mp4',
            'dmn_absolute': 'dmn_absolute_4000p.webm',
            
            // Bayesian statistics
            '1950-1980': '1950-1980.png',
            '1980-2000': '1980-2000.png',
            '2000-2017': '2000-2017.png',
            
            // Scientific disciplines
            'sch_1963-1997': 'sch_1963-1997.png',
            'sch_1998-2003': 'sch_1998-2003.png',
            'sch_2004-2009': 'sch_2004-2009.png',
            'sch_2010-2015': 'sch_2010-2015.png'
        };
        
        return mediaMappings[target] ? basePath + mediaMappings[target] : null;
    }

    // Gestion responsive des colonnes de texte
    initResponsiveColumns() {
        const textElements = document.querySelectorAll('.columns-text');
        
        if (textElements.length > 0) {
            // Observer les changements de taille
            const resizeObserver = new ResizeObserver(entries => {
                entries.forEach(entry => {
                    this.updateColumns(entry.target);
                });
            });
            
            textElements.forEach(element => {
                resizeObserver.observe(element);
                this.updateColumns(element);
            });
        }
    }

    updateColumns(element) {
        const width = element.offsetWidth;
        let columns = 1;
        
        if (width >= 1400) {
            columns = 4;
        } else if (width >= 1025) {
            columns = 3;
        } else if (width >= 769) {
            columns = 2;
        }
        
        element.style.columnCount = columns;
    }

    // Utilitaires
    static debounce(func, wait) {
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
}

// Initialiser le portfolio
const portfolio = new Portfolio();

// Exporter pour utilisation dans d'autres scripts si nécessaire
window.Portfolio = Portfolio;
