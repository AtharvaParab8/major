import React, { useState } from 'react';
import { Link as RouterLink, BrowserRouter } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemText,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';

function Navbar() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setDrawerOpen(open);
  };

  const navItems = (
    <>
      <Button color="inherit" variant="outlined" sx={{ mr: 1 }}>
        Login
      </Button>
      <Button color="primary" variant="contained">
        Sign Up
      </Button>
    </>
  );

  const drawerContent = (
    <List>
      <ListItem button component={RouterLink} to="/login">
        <ListItemText primary="Login" />
      </ListItem>
      <ListItem button component={RouterLink} to="/signup">
        <ListItemText primary="Sign Up" />
      </ListItem>
    </List>
  );

  return (
    <AppBar position="static" color="default" elevation={1}>
      <Toolbar>
        <Typography
          variant="h6"
          component={RouterLink}
          to="/"
          sx={{
            flexGrow: 1,
            textDecoration: 'none',
            color: 'inherit',
            fontWeight: 'bold',
          }}
        >
          Video Summarization
        </Typography>
        {isMobile ? (
          <>
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={toggleDrawer(true)}
            >
              <MenuIcon />
            </IconButton>
            <Drawer
              anchor="right"
              open={drawerOpen}
              onClose={toggleDrawer(false)}
            >
              {drawerContent}
            </Drawer>
          </>
        ) : (
          navItems
        )}
      </Toolbar>
    </AppBar>
  );
}

// Wrap the Navbar with BrowserRouter
export default function NavbarWithRouter() {
  return (
    <BrowserRouter>
      <Navbar />
    </BrowserRouter>
  );
}